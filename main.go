package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.com/e-XpertSolutions/go-iforest/iforest"
	"github.com/petar/GoMNIST"
)

func convertImages(imgs []GoMNIST.RawImage) [][]float64 {
	// images are 28x28 pixels
	xsize := 28 * 28

	imgArray := make([][]float64, len(imgs))
	for i := 0; i < len(imgs); i++ {
		imgArray[i] = make([]float64, xsize)
		for j := 0; j < xsize; j++ {
			imgArray[i][j] = float64(imgs[i][j])
		}
	}

	return imgArray
}

// Isolation trees do not actually require feature scaling, but since the Python code we will be comparing results to performs
// MinMaxScaler, we will perform as well.
// Struggled to find a package to automatically scale, so attempting to write my own
/*func featureScaling(data []float64) []float64 {


	return scaledData
}
*/

func writeGoResults(scores map[int]float64, labels []int) error {

	file, err := os.Create("results/goResults.csv")
	if err != nil {
		fmt.Println("Could not create file:", err)
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write the header
	if err := writer.Write([]string{"id", "iforestGoScore", "anomalyLabel"}); err != nil {
		fmt.Println("Error writing header:", err)
		return err
	}

	// loop through results and write to file
	for id, l := range labels {
		row := []string{
			fmt.Sprint(id),
			fmt.Sprintf("%.6f", scores[id]),
			fmt.Sprint(l),
		}
		if err := writer.Write(row); err != nil {
			fmt.Println("Error writing row:", err)
			return err
		}
	}
	return nil
}

func readPythonResults(pyscorefile, pyanomlabfile string) ([]float64, []int, error) {

	var pyscores []float64
	var pylabels []int

	// Open and read pythonScores file
	psfile, err := os.Open(pyscorefile)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil, nil, err
	}
	defer psfile.Close()

	// Create a new CSV reader.
	score_reader := csv.NewReader(psfile)

	// Read in all CSV data
	allScores, err := score_reader.ReadAll()
	if err != nil {
		fmt.Printf("Cannot read data from file: %v", err)
		return nil, nil, err
	}

	// Read and parse each line of the CSV file
	for i, row := range allScores {
		// Skip the header line
		if i == 0 {
			continue
		}
		ps, err := strconv.ParseFloat(row[1], 64)
		if err != nil {
			fmt.Printf("Error parsing to number: %v\n", err)
			return nil, nil, err
		}
		pyscores = append(pyscores, ps)
	}

	// Read and parse pythonAnomalyLabels file
	palfile, err := os.Open(pyanomlabfile)
	if err != nil {
		fmt.Println("Error opening file:", err)
		return nil, nil, err
	}
	defer palfile.Close()

	// Create a new CSV reader.
	pal_reader := csv.NewReader(palfile)

	// Read in all CSV data
	allALabels, err := pal_reader.ReadAll()
	if err != nil {
		fmt.Printf("Cannot read data from file: %v", err)
		return nil, nil, err
	}

	// Read and parse each line of the CSV file
	for i, row := range allALabels {
		// Skip the header line
		if i == 0 {
			continue
		}
		pal, err := strconv.Atoi(row[1])
		if err != nil {
			fmt.Printf("Error parsing to number: %v\n", err)
			return nil, nil, err
		}
		pylabels = append(pylabels, pal)
	}

	return pyscores, pylabels, nil

}

func compareResults(goscores map[int]float64, pyscores []float64, golabels, pylabels []int) (int, error) {

	// python anomaly labels are 1=normal and -1=anomaly
	// go anomaly labels are 0=normal and 1=anomaly
	var updated_pylabels []int
	for _, l := range pylabels {
		if l == 1 {
			updated_pylabels = append(updated_pylabels, 0)
		} else {
			updated_pylabels = append(updated_pylabels, 1)
		}

	}

	// Write out all results
	file, err := os.Create("results/comparedResults.csv")
	if err != nil {
		fmt.Println("Could not create file:", err)
		return -1, err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write the header
	if err := writer.Write([]string{"id", "goScore", "pythonScore", "goAnomalyLabel", "pythonAnomalyLabel"}); err != nil {
		fmt.Println("Error writing header:", err)
		return -1, err
	}

	// count number of mislabeled between python and go results
	var mislabeled int

	// loop through results and write to file
	for id, gol := range golabels {
		row := []string{
			fmt.Sprint(id),
			fmt.Sprintf("%.6f", goscores[id]),
			fmt.Sprintf("%.6f", pyscores[id]),
			fmt.Sprint(gol),
			fmt.Sprint(updated_pylabels[id]),
		}
		if err := writer.Write(row); err != nil {
			fmt.Println("Error writing row:", err)
			return -1, err
		}

		if gol != updated_pylabels[id] {
			mislabeled++
		}
	}

	return mislabeled, nil
}

func main() {
	// Isolation forests will be constructed with training images only
	_, _, imgs, err := GoMNIST.ReadImageFile("./data/train-images-idx3-ubyte.gz")
	if err != nil {
		panic(err)
	}

	// input data for iforest must be loaded into two dimensional array of the type float64
	imgArray := convertImages(imgs)

	// input parameters
	treesNumber := 100   // similar to n_estimators
	subsampleSize := 256 // similar to max_samples
	// Contamination: Python code uses "auto." Scikit-learn documentation says "If ‘auto’, the threshold is determined as in the original paper."
	// Did not have time to track down what the actual formula was, so just selecting value arbitrarily.
	outliersRatio := 0.1

	//model initialization
	forest := iforest.NewForest(treesNumber, subsampleSize, outliersRatio)

	//training stage - creating trees
	forest.Train(imgArray)

	//testing stage - finding anomalies
	forest.Test(imgArray)

	//after testing it is possible to access anomaly scores, anomaly bound
	// and labels for the input dataset

	threshold := forest.AnomalyBound
	fmt.Println("Threshold:", threshold)
	anomalyScores := forest.AnomalyScores
	// print count of anomalies detected
	anomalyLabels := forest.Labels // 0 is normal, 1 is anomaly
	anomalyCount := 0
	for _, lab := range anomalyLabels {
		if lab == 1 {
			anomalyCount++
		}
	}
	fmt.Println("Number of anomalies:", anomalyCount)

	err = writeGoResults(anomalyScores, anomalyLabels)
	if err != nil {
		fmt.Println("Error writing results:", err)
	}

	// read in python results and compare to go results
	pyscores, pylabels, err := readPythonResults("./results/pythonScores.csv", "./results/pythonAnomalyLabels.csv")

	mislabeled, err := compareResults(anomalyScores, pyscores, anomalyLabels, pylabels)

	fmt.Println("Number of anomaly labels that don't match between Go and Python results:", mislabeled)

	max_score := 0.0
	min_score := 1.0
	for _, sv := range anomalyScores {
		if sv > max_score {
			max_score = sv
		}
		if sv < min_score {
			min_score = sv
		}
	}
	fmt.Println(min_score)
	fmt.Println(max_score)

}
