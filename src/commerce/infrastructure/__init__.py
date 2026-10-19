"""基础设施层：实现 domain 定义的仓储 Protocol，并承载工程化横切能力。

包含：业务库连接（db/）、仓储实现（repositories/）、缓存（cache.py）、
进程内调度器（scheduler.py）、演示种子数据（seed.py）。
"""

from __future__ import annotations

__all__ = ["__doc__"]
