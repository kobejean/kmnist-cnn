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
func readFile(_ filename: String) -> [UInt8] {
    let data = Python.open(filename, "rb").read()
    return Array(numpyArray: np.frombuffer(data, dtype: np.uint8))!
}

/// Reads KMNIST images and labels from specified file paths.
@inline(never)
func readKMNIST(imagesFile: String, labelsFile: String, height: Int32 = 28, width: Int32 = 28, channels: Int32 = 1) -> [(images: Tensor<Float>, labels: Tensor<Int32>)] {
    print("Reading data.")
    var images = readFile(imagesFile).dropFirst(16).map { Float($0) / 255 }
    var labels = readFile(labelsFile).dropFirst(8).map { Int32($0) }
    let epochSize = labels.count
    let batchSize = 100
    let batchCount = (epochSize + batchSize - 1) / batchSize
    
    let imageBatchSize = batchSize * Int(height * width * channels)
    let labelBatchSize = batchSize
    
    print("Constructing batches.")
    var dataset = [(Tensor<Float>, Tensor<Int32>)]()
    for i in 0..<Int(batchCount) {
        let imageBatchRangeStart = i * imageBatchSize
        let imageBatchRangeEnd = min(images.count, imageBatchRangeStart + imageBatchSize)
        let labelBatchRangeStart = i * labelBatchSize
        let labelBatchRangeEnd = min(labels.count, labelBatchRangeStart + labelBatchSize)
        let currentBatchSize = Int32(labelBatchRangeEnd - labelBatchRangeStart)
        
        let imageBatch = Array(images[imageBatchRangeStart..<imageBatchRangeEnd])
        let imageBatchShaped = Tensor(shape: [currentBatchSize, height, width, channels], scalars: imageBatch)
        let labelBatch = Array(labels[labelBatchRangeStart..<labelBatchRangeEnd])
        let labelBatchShaped = Tensor(shape: [currentBatchSize], scalars: labelBatch)
        dataset.append((imageBatchShaped, labelBatchShaped))
    }
    return dataset
}
