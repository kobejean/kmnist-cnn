//
//  CNNModel.swift
//  kmnist
//
//  Created by Jean Flaherty on 12/8/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow
import Foundation

/// Parameters of an MNIST classifier.
@usableFromInline
struct CNNParameters : Parameterized {
//    public typealias TangentVector = CNNParameters
//    public typealias CotangentVector = CNNParameters
    
    @TFParameter @usableFromInline var k1 = Tensor<Float>(glorotUniform: [5, 5, 1, 32])
    @TFParameter @usableFromInline var k2 = Tensor<Float>(glorotUniform: [5, 5, 32, 64])
    @TFParameter @usableFromInline var w3 = Tensor<Float>(glorotUniform: [Int32(7*7*64), 10])
    @TFParameter @usableFromInline var b3 = Tensor<Float>(zeros: [10])
    
//    func moved(along direction: CNNParameters) -> CNNParameters {
//        return self + direction
//    }
//
//    func tangentVector(from cotangent: CNNParameters) -> CNNParameters {
//        return cotangent
//    }
}

@usableFromInline
func lossAndGradient(_ x: Tensor<Float>, _ y_i: Tensor<Int32>, _ θ: CNNParameters, returnGradient: Bool = true) -> (Float, CNNParameters) {
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
    let p: Tensor<Float> = e_i(y_i, 10)
    let H = ∑(p * -log(q))
    let H_total = μ(H)
//
//    guard returnGradient else { return (H_total, θ) }
//
//    // Backpropagation
//    let dz3 = q - p
//    let dw3 = flat⊺ • dz3
//    let db3 = dz3.sum(squeezingAxes: 0)
//    let dflat = dz3 • θ.w3⊺
//    let dm2 = pullback(at: m2, in: { $0.reshaped(to: flat.shape) })(dflat)
//    let dh2 = pullback(at: h2, in: { $0.maxPooled(kernelSize: kernelSize5, strides: strides2, padding: .same) })(dm2)
//    let dc2 = pullback(at: c2, in: { relu($0) })(dh2)
//    let (dm1, dk2) = pullback(at: m1, θ.k2, in: { $0.convolved2D(withFilter: $1, strides: strides1, padding: .same) })(dc2)
//    let dh1 = pullback(at: h1, in: { $0.maxPooled(kernelSize: kernelSize5, strides: strides2, padding: .same) })(dm1)
//    let dc1 = pullback(at: c1, in: { relu($0) })(dh1)
//    let dk1 = pullback(at: θ.k1, in: { x.convolved2D(withFilter: $0, strides: strides1, padding: .same) })(dc1)
//
//    let dθ = CNNParameters(k1: dk1, k2: dk2, w3: dw3, b3: db3)
    return (H_total, θ)
}

/// Starts training on KMNIST
func startTrainingOnKMNIST() {
    let dataPath = "usr/share/man/man1/kmnist/Data/"
    let trainingImagesFile = dataPath + "kmnist-train-imgs.npz"
    let trainingLabelsFile = dataPath + "kmnist-train-labels.npz"
    let testingImagesFile = dataPath + "kmnist-test-imgs.npz"
    let testingLabelsFile = dataPath + "kmnist-test-labels.npz"

    let batchSize = 128
    var θ = CNNParameters()
    let α: Float = 0.001
    let βm: Float = 0.9
    let βv: Float = 0.999
    let ϵ: Float = 1e-08
//    let adam = AdamOptimizer(θ,α,βm,βv,ϵ)

    print("Load training data.")
    var trainingDataset = readKMNIST(imagesFile: trainingImagesFile, labelsFile: trainingLabelsFile, batchSize: batchSize)
    print("Load validation data.")
    let testingBatch = readKMNIST(imagesFile: testingImagesFile, labelsFile: testingLabelsFile)[0]

    printDividerLine()
    for epochNumber in 0..<256 {
        trainingDataset.shuffle()
        for (batchNumber, batch) in trainingDataset.enumerated() {
            let x = batch.images
            let y_i = batch.labels

            let (H_total, dθ) = lossAndGradient(x, y_i, θ)
//            adam.optimize(&θ,dθ)

            if batchNumber % 10 == 0 {
                printTrainingLoss(epochNumber: epochNumber, batchNumber: batchNumber, H_total: H_total)
            }
            if batchNumber != 0, batchNumber % 200 == 0 || batchNumber == trainingDataset.count-1 {
                printDividerLine()
                let (H_test, _) = lossAndGradient(testingBatch.images, testingBatch.labels, θ, returnGradient: false)
                printValidationLoss(epochNumber: epochNumber, batchNumber: batchNumber, H_test: H_test)
                printDividerLine()
            }
        }
    }
}
