//
//  Dataset.swift
//  KMNIST
//
//  Created by Jean Flaherty on 12/8/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import Python
import TensorFlow

/// Reads numpy file
func readFile<Scalar: NumpyScalarCompatible>(_ filename: String) -> [Scalar] {
    let data = np.load(filename)["arr_0"]
    return Array(numpyArray: data.flatten().astype(Scalar.ctype))!
}

/// Reads KMNIST images and labels from specified file paths.
@inline(never)
func readKMNIST(imagesFile: String, labelsFile: String, batchSize: Int? = nil, height: Int32 = 28, width: Int32 = 28, channels: Int32 = 1) -> [(images: Tensor<Float>, labels: Tensor<Int32>)] {
    print("Reading data.")
    let pixels: [Float] = readFile(imagesFile)
    let labels: [Int32] = readFile(labelsFile)
    let epochSize = labels.count
    let batchSize = batchSize ?? epochSize
    let batchCount = (epochSize + batchSize - 1) / batchSize
    
    let imageBatchSize = batchSize * Int(height * width * channels)
    let labelBatchSize = batchSize
    
    print("Constructing batches.")
    var dataset = [(Tensor<Float>, Tensor<Int32>)]()
    for i in 0..<Int(batchCount) {
        let imageBatchRangeStart = i * imageBatchSize
        let imageBatchRangeEnd = min(pixels.count, imageBatchRangeStart + imageBatchSize)
        let labelBatchRangeStart = i * labelBatchSize
        let labelBatchRangeEnd = min(labels.count, labelBatchRangeStart + labelBatchSize)
        let currentBatchSize = Int32(labelBatchRangeEnd - labelBatchRangeStart)
        
        let imageBatch = Array(pixels[imageBatchRangeStart..<imageBatchRangeEnd])
        let imageBatchShaped = Tensor(shape: [currentBatchSize, height, width, channels], scalars: imageBatch) / 255
        let labelBatch = Array(labels[labelBatchRangeStart..<labelBatchRangeEnd])
        let labelBatchShaped = Tensor(shape: [currentBatchSize], scalars: labelBatch)
        dataset.append((imageBatchShaped, labelBatchShaped))
    }
    return dataset
}
