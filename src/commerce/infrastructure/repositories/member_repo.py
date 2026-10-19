"""会员仓储。口令哈希与盐只在此处进出库，绝不出现在对外 DTO 里。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...domain.models import Address, Member, MemberLevel
from ..db.models import AddressRecord, MemberRecord


class SqlMemberRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _load(self, record: MemberRecord) -> Member:
        rows = (
            (await self._session.execute(select(AddressRecord).where(AddressRecord.member_id == record.member_id)))
            .scalars()
            .all()
        )
        return Member(
            member_id=record.member_id,
            email=record.email,
            nickname=record.nickname or "",
            level=MemberLevel(record.level),
            salt_hex=record.salt_hex,
            hash_hex=record.hash_hex,
            created_at=record.created_at,
            addresses=[
                Address(
                    address_id=row.address_id,
                    member_id=row.member_id,
                    receiver=row.receiver,
                    phone=row.phone,
                    country=row.country,
                    city=row.city,
                    line1=row.line1,
                    is_default=row.is_default,
                )
                for row in rows
            ],
        )

    async def get(self, member_id: str) -> Member | None:
        record = (
            await self._session.execute(select(MemberRecord).where(MemberRecord.member_id == member_id))
        ).scalar_one_or_none()
        return await self._load(record) if record else None

    async def by_email(self, email: str) -> Member | None:
        record = (
            await self._session.execute(select(MemberRecord).where(MemberRecord.email == email.strip().lower()))
        ).scalar_one_or_none()
        return await self._load(record) if record else None

    async def save(self, member: Member) -> None:
        record = (
            await self._session.execute(select(MemberRecord).where(MemberRecord.member_id == member.member_id))
        ).scalar_one_or_none()
        if record is None:
            record = MemberRecord(member_id=member.member_id, email=member.email.lower())
            self._session.add(record)
        record.nickname = member.nickname
        record.level = member.level.value
        record.salt_hex = member.salt_hex
        record.hash_hex = member.hash_hex
        await self._session.flush()

    async def save_address(self, member_id: str, address: Address) -> None:
        record = (
            await self._session.execute(select(AddressRecord).where(AddressRecord.address_id == address.address_id))
        ).scalar_one_or_none()
        if record is None:
            record = AddressRecord(address_id=address.address_id, member_id=member_id)
            self._session.add(record)
        record.receiver = address.receiver
        record.phone = address.phone
        record.country = address.country
        record.city = address.city
        record.line1 = address.line1
        record.is_default = address.is_default
        if address.is_default:
            # 默认地址全局唯一：切换默认时要先把旧的取消，否则会出现两个默认地址
            others = (
                (
                    await self._session.execute(
                        select(AddressRecord).where(
                            AddressRecord.member_id == member_id, AddressRecord.address_id != address.address_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            for other in others:
                other.is_default = False
        await self._session.flush()


__all__ = ["SqlMemberRepository"]
