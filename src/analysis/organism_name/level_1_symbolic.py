import os
from datetime import datetime

from root import from_root
from src.analysis.symbolic_trials.level_1_symbolic_trials import baseline, \
    heuristical, better_baseline, heuristical_with_test_outcome
from src.modules.db import extract


SAVE_TO = from_root("results\\organism_name\\level_1_symbolic")


def main():
    df = extract(from_root("sql\\organism_name\\level_1_symbolic.sql"))

    baseline(df, os.path.join(SAVE_TO, "baseline"))
    better_baseline(df, os.path.join(SAVE_TO, "better_baseline"))
    heuristical(df, os.path.join(SAVE_TO, "heuristical"))
    heuristical_with_test_outcome(
        df, os.path.join(SAVE_TO, "heuristical_with_test_outcome"))


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    main()

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
