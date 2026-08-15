from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared base class for all ORM models. Having one Base means
    Base.metadata.create_all() (in database.py) knows about every table
    in the app, no matter which file it's defined in."""
    pass
