//
//  KMNIST.swift
//  KMNIST
//
//  Created by Jean Flaherty on 12/8/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow
import Foundation

/// Parameters of an MNIST classifier.
@usableFromInline
struct KMNISTParameters : ParameterGroup {
    var k1 = Tensor<Float>(glorotUniform: [5, 5, 1, 32])
    var k2 = Tensor<Float>(glorotUniform: [5, 5, 32, 64])
    var w3 = Tensor<Float>(glorotUniform: [Int32(7*7*64), 10])
    var b3 = Tensor<Float>(zeros: [10])
}

@usableFromInline @inline(never)
func trainingStep(_ x: Tensor<Float>, _ y_i: Tensor<Int32>, _ θ: KMNISTParameters, returnGradient: Bool = true) -> (Float, KMNISTParameters) {
    let strides1: (Int32, Int32, Int32, Int32) = (1, 1, 1, 1)
    let strides2: (Int32, Int32, Int32, Int32) = (1, 2, 2, 1)
    let kernelSize5: (Int32, Int32, Int32, Int32) = (1, 5, 5, 1)
    
    // Inferance
    let c1 = x.convolved2D(withFilter: θ.k1, strides: strides1, padding: .same)
    let h1 = relu(c1)
    let m1 = h1.maxPooled(kernelSize: kernelSize5, strides: strides2, padding: .same)
    let c2 = m1.convolved2D(withFilter: θ.k2, strides: strides1, padding: .same)
    let h2 = relu(c2)
    let m2 = h2.maxPooled(kernelSize: kernelSize5, strides: strides2, padding: .same)
    let flat = m2.reshaped(to: [-1, Int32(7*7*64)])
    let z3 = flat • θ.w3 + θ.b3
    let h3 = softmax(z3, alongAxis: -1)
    
    // Evaluation
    let q = h3
    let p = e_i(y_i, 10)
    let H = -Σ(p * log(q))
    let H_total = μ(H)
    
    guard returnGradient else { return (H_total, θ) }
    
    // Backpropagation
    let dz3 = q - p
    let dw3 = flat⊺ • dz3
    let db3 = Σ(dz3,0)
    let dflat = dz3 • θ.w3⊺
    let dm2 = #adjoint(Tensor.reshaped)(m2)(toShape: flat.shapeTensor, originalValue: flat, seed: dflat)
    let dh2 = #adjoint(Tensor.maxPooled)(h2)(kernelSize: kernelSize5, strides: strides2, padding: .same, originalValue: m2, seed: dm2)
    let dc2 = #adjoint(relu)(c2, originalValue: h2, seed: dh2)
    let (dm1, dk2) = #adjoint(Tensor<Float>.convolved2D)(m1)(filter: θ.k2, strides: strides1, padding: .same, originalValue: c2, seed: dc2)
    let dh1 = #adjoint(Tensor.maxPooled)(h1)(kernelSize: kernelSize5, strides: strides2, padding: .same, originalValue: m1, seed: dm1)
    let dc1 = #adjoint(relu)(c1, originalValue: h1, seed: dh1)
    let (_, dk1) = #adjoint(Tensor<Float>.convolved2D)(x)(filter: θ.k1, strides: strides1, padding: .same, originalValue: c1, seed: dc1)
    
    let dθ = KMNISTParameters(k1: dk1, k2: dk2, w3: dw3, b3: db3)
    return (H_total, dθ)
}

/// Starts training on KMNIST
func startTrainingOnKMNIST() {
    let dataPath = "usr/share/man/man1/KMNIST/Data/"
    let trainingImagesFile = dataPath + "kmnist-train-imgs.npz"
    let trainingLabelsFile = dataPath + "kmnist-train-labels.npz"
    let testingImagesFile = dataPath + "kmnist-test-imgs.npz"
    let testingLabelsFile = dataPath + "kmnist-test-labels.npz"
    let batchSize = 128
    var θ = KMNISTParameters()
    let η: Float = 0.00075

    let trainingDataset = readKMNIST(imagesFile: trainingImagesFile, labelsFile: trainingLabelsFile, batchSize: batchSize)
    let testingBatch = readKMNIST(imagesFile: testingImagesFile, labelsFile: testingLabelsFile)[0]
    
    printDividerLine()
    for epochNumber in 0..<16 {
        for (batchNumber, batch) in trainingDataset.enumerated() {
            let x = batch.images
            let y_i = batch.labels
            let (H_total, dθ) = trainingStep(x, y_i, θ)
            
            // Update gradients
            θ.update(withGradients: dθ) { (θ_k, dθ_k) in θ_k -= η * dθ_k }
            
            if batchNumber % 10 == 0 {
                printTrainingLoss(epochNumber: epochNumber, batchNumber: batchNumber, H_total: H_total)
            }
            if batchNumber != 0, batchNumber % 200 == 0 || batchNumber == trainingDataset.count-1 {
                printDividerLine()
                let (H_test, _) = trainingStep(testingBatch.images, testingBatch.labels, θ, returnGradient: false)
                printValidationLoss(epochNumber: epochNumber, batchNumber: batchNumber, H_test: H_test)
                printDividerLine()
            }
        }
    }
}
