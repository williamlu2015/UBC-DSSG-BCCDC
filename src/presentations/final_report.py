import os

import matplotlib.pyplot as plt

from datetime import datetime

from root import from_root
from src.util.io import write_matplotlib_figure


SAVE_TO = from_root("images\\final_report")


def main():
    test_performed_labeled_pie_chart()
    test_performed_classes_pie_chart()
    test_outcome_labeled_pie_chart()
    test_outcome_classes_pie_chart()
    organism_name_labeled_pie_chart()
    level_1_classes_pie_chart()


def test_performed_labeled_pie_chart():
    labels = ["Labeled"]
    sizes = [362863]

    pie = plt.pie(sizes, autopct="%1.2f%%")
    plt.axis("equal")

    plt.title("Proportion of dataset with Test Performed labels")
    plt.legend(pie[0], labels)

    write_matplotlib_figure(
        os.path.join(SAVE_TO, "test_performed_labeled_prop.png"), plt)


def test_performed_classes_pie_chart():
    labels = ["Yes", "No"]
    sizes = [340476, 22169]

    pie = plt.pie(sizes, autopct="%1.2f%%")
    plt.axis("equal")

    plt.title("Class breakdown for Test Performed")
    plt.legend(pie[0], labels)

    write_matplotlib_figure(
        os.path.join(SAVE_TO, "test_performed_classes_prop.png"), plt)


def test_outcome_labeled_pie_chart():
    labels = ["Labeled", "Unlabeled"]
    sizes = [115355, 247508]
    explode = (0.1, 0)

    pie = plt.pie(sizes, explode=explode, autopct="%1.2f%%")
    plt.axis("equal")

    plt.title("Proportion of dataset with Test Outcome labels")
    plt.legend(pie[0], labels)

    write_matplotlib_figure(
        os.path.join(SAVE_TO, "test_outcome_labeled_prop.png"), plt)


def test_outcome_classes_pie_chart():
    labels = ["Positive", "Negative", "Indeterminate", "Missing"]
    sizes = [19249, 15268, 1453, 79385]

    pie = plt.pie(sizes, autopct="%1.2f%%")
    plt.axis("equal")

    plt.title("Class breakdown for Test Outcome")
    plt.legend(pie[0], labels)

    write_matplotlib_figure(
        os.path.join(SAVE_TO, "test_outcome_classes_prop.png"), plt)


def organism_name_labeled_pie_chart():
    labels = ["Labeled", "Unlabeled"]
    sizes = [40289, 322574]
    explode = (0.1, 0)

    pie = plt.pie(sizes, explode=explode, autopct="%1.2f%%")
    plt.axis("equal")

    plt.title("Proportion of dataset with Organism Name labels")
    plt.legend(pie[0], labels)

    write_matplotlib_figure(
        os.path.join(SAVE_TO, "organism_name_labeled_prop.png"), plt)


def level_1_classes_pie_chart():
    sizes = [
        18354, 8552, 2646, 1477, 1248, 1245, 513, 466, 398, 360, 352, 298, 270,
        269, 252, 244, 232, 224, 192, 158, 151, 135, 108, 75, 68, 45, 38
    ]
    labels = ["Not Found"] + ["" for _ in range(26)]
    explode = tuple([0.1] + [0 for _ in range(26)])

    plt.pie(sizes, explode=explode, labels=labels, labeldistance=0.6)
    plt.axis("equal")

    plt.title("Class breakdown for Level 1")

    write_matplotlib_figure(
        os.path.join(SAVE_TO, "level_1_classes_prop.png"), plt)


if __name__ == "__main__":
    print("Started executing script.\n")
    start_time = datetime.now()

    main()

    print(f"\nExecution time: {datetime.now() - start_time}")
    print("Finished executing script.")
