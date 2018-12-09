# Training A Convolutional Neural Network On KMNIST

Download the dataset here: [KMNIST Dataset](https://github.com/rois-codh/kmnist)

#### Model
```swift
struct KMNISTParameters : ParameterGroup {
    var k1 = Tensor<Float>(glorotUniform: [5, 5, 1, 32])
    var k2 = Tensor<Float>(glorotUniform: [5, 5, 32, 64])
    var w3 = Tensor<Float>(glorotUniform: [3136, 10])
    var b3 = Tensor<Float>(zeros: [10])
}
```
#### Inference
```swift
let c1 = x.convolved2D(withFilter: θ.k1, strides: (1, 1, 1, 1), padding: .same)
let h1 = relu(c1)
let m1 = h1.maxPooled(kernelSize: (1, 5, 5, 1), strides: (1, 2, 2, 1), padding: .same)
let c2 = m1.convolved2D(withFilter: θ.k2, strides: (1, 1, 1, 1), padding: .same)
let h2 = relu(c2)
let m2 = h2.maxPooled(kernelSize: (1, 5, 5, 1), strides: (1, 2, 2, 1), padding: .same)
let flat = m2.reshaped(to: [-1, 3136])
let z3 = flat • θ.w3 + θ.b3
let h3 = softmax(z3, alongAxis: -1)
```
#### Loss
```swift
let q = h3
let p = e_i(y_i, 10)
let H = -Σ(p * log(q))
let H_total = μ(H)
```
#### Backpropagation
```swift
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
```
