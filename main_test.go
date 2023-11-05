package main

import (
	"testing"

	"github.com/petar/GoMNIST"
)

func TestReadData(t *testing.T) {
	_, _, imgs, err := GoMNIST.ReadImageFile("./data/train-images-idx3-ubyte.gz")
	if err != nil {
		t.Errorf("No data found")
	}

	if len(imgs) != 60000 {
		t.Errorf("Expected 60000 images, got %d", len(imgs))
	}

	if len(imgs[0]) != 784 {
		t.Errorf("Expected 784 pixels per image, got %d", len(imgs[0]))
	}
}
