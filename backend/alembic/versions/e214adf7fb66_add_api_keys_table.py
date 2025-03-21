"""add_api_keys_table

Revision ID: e214adf7fb66
Revises: 5be054bd6587
Create Date: 2024-01-20 13:24:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e214adf7fb66'
down_revision: Union[str, None] = '5be054bd6587'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
