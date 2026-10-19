"""会员凭证：口令哈希 + 会员 token 签发/校验。

放在 domain 而不是 infrastructure，是因为它**只依赖 stdlib**（hashlib / hmac / base64），
换数据库、换框架都不影响；而把它放进 interface 则会让「口令策略」这种业务规则
散落在 HTTP 层，无法被单测直接覆盖。

三个安全决策（面试追问「你自己做登录会不会出事」的答案）：

1. **口令不落明文、也不落快速哈希**：用 PBKDF2-HMAC-SHA256 + 每用户独立盐 + 高迭代，
   直接 `sha256(password)` 会被彩虹表与 GPU 秒破；
2. **比较用常量时间**：`hmac.compare_digest`，避免通过响应耗时侧信道逐字节猜哈希；
3. **token 是「签名的载荷」而不是随机串**：把 member_id / 过期时间放进载荷并用
   HMAC 签名，服务端无需存储会话即可校验，且能天然带过期时间；
   载荷里**不放任何权限信息**（权限一律服务端查库），否则就成了可被篡改的 cookie。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any

from .errors import MemberAuthenticationError, WeakPasswordError

_PBKDF2_ROUNDS = 120_000
_SALT_BYTES = 16
_MIN_PASSWORD_LEN = 8


def hash_password(password: str, *, salt: bytes | None = None) -> tuple[str, str]:
    """返回 (salt_hex, hash_hex)。盐随用户存库，绝不复用全局盐。"""
    if len(password or "") < _MIN_PASSWORD_LEN:
        raise WeakPasswordError(f"口令长度至少 {_MIN_PASSWORD_LEN} 位", detail={"min_len": _MIN_PASSWORD_LEN})
    raw_salt = salt or secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), raw_salt, _PBKDF2_ROUNDS)
    return raw_salt.hex(), derived.hex()


def verify_password(password: str, *, salt_hex: str, hash_hex: str) -> bool:
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _PBKDF2_ROUNDS)
    return hmac.compare_digest(derived.hex(), hash_hex)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64url(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


@dataclass(frozen=True, slots=True)
class MemberToken:
    member_id: str
    expires_at: int

    def to_dict(self) -> dict[str, Any]:
        return {"member_id": self.member_id, "expires_at": self.expires_at}


class MemberTokenSigner:
    """无状态会员 token 签发/校验（HMAC-SHA256）。

    格式：`<b64url(payload_json)>.<b64url(hmac)>`。
    服务端不存会话表，因此水平扩容时不需要共享 session；代价是**无法单点吊销**，
    生产上会叠加「token 版本号」解决（改密码即递增 version，旧 token 立刻失效）。
    """

    def __init__(self, secret: str, *, ttl_s: int = 86400) -> None:
        if not secret:
            raise MemberAuthenticationError("会员 token 密钥未配置")
        self._secret = secret.encode("utf-8")
        self._ttl_s = ttl_s

    def issue(self, member_id: str, *, now: int | None = None) -> tuple[str, int]:
        issued = now or int(time.time())
        expires = issued + self._ttl_s
        payload = json.dumps({"sub": member_id, "exp": expires}, separators=(",", ":"), sort_keys=True)
        body = _b64url(payload.encode("utf-8"))
        signature = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest()
        return f"{body}.{_b64url(signature)}", expires

    def verify(self, token: str, *, now: int | None = None) -> MemberToken:
        if not token or token.count(".") != 1:
            raise MemberAuthenticationError("会员令牌格式非法")
        body, signature = token.split(".")
        expected = hmac.new(self._secret, body.encode("ascii"), hashlib.sha256).digest()
        try:
            provided = _unb64url(signature)
        except Exception as exc:  # noqa: BLE001
            raise MemberAuthenticationError("会员令牌签名不可解析") from exc
        # 签名先比，再看载荷：顺序反了会给攻击者一个「先伪造载荷再猜签名」的窗口
        if not hmac.compare_digest(expected, provided):
            raise MemberAuthenticationError("会员令牌签名不匹配")
        try:
            payload = json.loads(_unb64url(body))
        except Exception as exc:  # noqa: BLE001
            raise MemberAuthenticationError("会员令牌载荷不可解析") from exc
        expires = int(payload.get("exp", 0))
        if expires <= (now or int(time.time())):
            raise MemberAuthenticationError("会员令牌已过期")
        subject = str(payload.get("sub") or "")
        if not subject:
            raise MemberAuthenticationError("会员令牌缺少主体")
        return MemberToken(member_id=subject, expires_at=expires)


__all__ = [
    "MemberToken",
    "MemberTokenSigner",
    "hash_password",
    "verify_password",
]
