from __future__ import annotations

from sqlalchemy import func, select

from euromillions.combinations import build_main_combinations, build_star_combinations
from euromillions.db import begin, create_all_tables, create_db_engine
from euromillions.schema import main_combinations, star_combinations


def test_combination_counts(tmp_path) -> None:
    engine = create_db_engine(str(tmp_path / "t.sqlite"))
    create_all_tables(engine)
    with begin(engine) as conn:
        build_main_combinations(conn, batch_size=100000)
        build_star_combinations(conn)
        mains = conn.execute(select(func.count()).select_from(main_combinations)).scalar_one()
        stars = conn.execute(select(func.count()).select_from(star_combinations)).scalar_one()
    assert mains == 2_118_760
    assert stars == 66
