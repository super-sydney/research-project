import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # csvs = [x for x in os.listdir("./results") if x.endswith(".csv")]
    csvs = ["Zernike", "BesselFourier", "Legendre", "Tchebichef", "Gabor", "GaborZernike",
            "GaborLegendre"]

    # Load the results
    results = [pd.read_csv(f"./results/both/{csv}Both.csv") for csv in csvs]
    results.append(pd.read_csv("./results/GaborZernikemaybe.csv"))
    csvs.append("GaborZernikemaybe.csv")

    # for each result, calculate average precision and recall for every number of retrieved images
    avg_precisions = []
    avg_recalls = []
    for result in results:
        average_precision = []
        average_recall = []

        for i in range(1, max(result["number_retrieved"])):
            precision = result[result["number_retrieved"] == i]["precision"].mean()
            recall = result[result["number_retrieved"] == i]["recall"].mean()

            average_precision.append(precision)
            average_recall.append(recall)

        avg_precisions.append(average_precision)
        avg_recalls.append(average_recall)

    # define colors using hex codes
    colors = ["#002ebf", "#9a00d0", "#ff0080", "#d28000", "#00e5ff", "#54ff00", "#dcd600", "#ff0000"]
    for i in range(len(avg_precisions)):
        # set color of line
        plt.plot(avg_recalls[i], avg_precisions[i], label=csvs[i][:-4], color=colors[i])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.xticks([i / 10 for i in range(11)])
    plt.yticks([i / 10 for i in range(11)])
    plt.xlabel("Average Recall")
    plt.ylabel("Average Precision")
    plt.title("Precision-Recall Curve")
    # plt.legend()
    # plt.savefig("both.png")
    plt.show()
