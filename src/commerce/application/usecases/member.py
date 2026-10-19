"""会员用例：注册 / 登录 / 资料 / 收货地址。

四个安全要点（面试追问「你自己写登录会踩什么坑」）：
1) **注册与登录分离**：注册检查邮箱唯一性并落库，登录只做「取盐 → 验哈希」，
   两者共用的是口令哈希函数，不是流程 —— 混在一起会让「登录失败」变成「邮箱不存在」的信息泄露；
2) **失败信息不区分「邮箱不存在」与「口令错误」**：都返回同一句 `member_unauthenticated`，
   否则接口就成了账号枚举器；
3) **token 里不放权限**：只放 member_id 与过期时间并签名，等级（NORMAL/SILVER/GOLD）
   一律服务端查库 —— 折扣率若能从 token 里读，改一个 base64 就能自封 GOLD；
4) **默认地址唯一**：切换默认时必须把旧的取消，否则「两个默认地址」会让结算用错地址。
"""

from __future__ import annotations

from src.infra.core.errors import NotFoundError

from ...domain.errors import MemberAuthenticationError, MemberExistsError
from ...domain.ids import new_id
from ...domain.models import Address, Member, MemberLevel
from ...domain.security import MemberTokenSigner, hash_password
from ..ports import UowFactory


class MemberUseCase:
    def __init__(self, uow_factory: UowFactory, signer: MemberTokenSigner) -> None:
        self._uow = uow_factory
        self._signer = signer

    async def register(self, *, email: str, password: str, nickname: str = "") -> Member:
        normalized = email.strip().lower()
        salt_hex, hash_hex = hash_password(password)  # 弱口令在这里就被拒（weak_password）
        async with self._uow() as uow:
            if await uow.members.by_email(normalized) is not None:
                raise MemberExistsError(f"邮箱已注册: {normalized}", detail={"email": normalized})
            member = Member(
                member_id=new_id("u", length=10),
                email=normalized,
                nickname=nickname.strip(),
                level=MemberLevel.NORMAL,
                salt_hex=salt_hex,
                hash_hex=hash_hex,
            )
            await uow.members.save(member)
            await uow.commit()
        return member

    async def login(self, *, email: str, password: str) -> tuple[Member, str, int]:
        async with self._uow() as uow:
            member = await uow.members.by_email(email.strip().lower())
            await uow.commit()
        if member is None or not member.verify_password(password):
            # 刻意合并「不存在」与「口令错」：不给账号枚举留下可区分的响应
            raise MemberAuthenticationError("邮箱或口令不正确")
        token, expires_at = self._signer.issue(member.member_id)
        return member, token, expires_at

    async def profile(self, member_id: str) -> Member:
        async with self._uow() as uow:
            member = await uow.members.get(member_id)
            await uow.commit()
        if member is None:
            raise NotFoundError(f"会员不存在: {member_id}", detail={"member_id": member_id})
        return member

    async def add_address(
        self,
        member_id: str,
        *,
        receiver: str,
        phone: str,
        country: str,
        city: str,
        line1: str,
        is_default: bool = False,
    ) -> Address:
        address = Address(
            address_id=new_id("addr"),
            member_id=member_id,
            receiver=receiver,
            phone=phone,
            country=country,
            city=city,
            line1=line1,
            is_default=is_default,
        )
        async with self._uow() as uow:
            await uow.members.save_address(member_id, address)
            await uow.commit()
        return address

    def verify_token(self, token: str) -> str:
        """校验会员 token 并返回 member_id。签名/过期由 domain 的无状态签章器判定。"""
        return self._signer.verify(token).member_id


__all__ = ["MemberUseCase"]
