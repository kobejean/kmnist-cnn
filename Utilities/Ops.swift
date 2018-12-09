//
//  Ops.swift
//  Iris
//
//  Created by Jean Flaherty on 12/2/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

@inlinable @inline(__always)
public func oneHot<Scalar: TensorFlowScalar>(indices: Tensor<Int32>, depth: Int32, onValue: Scalar, offValue: Scalar, axis: Int64? = nil) -> Tensor<Scalar> {
    let depthTensor = Tensor<Int32>(depth)
    let onValueTensor = Tensor<Scalar>(onValue)
    let offValueTensor = Tensor<Scalar>(offValue)
    if let axis = axis {
        return Raw.oneHot(indices: indices, depth: depthTensor, onValue: onValueTensor, offValue: offValueTensor, axis: axis)
    }
    return Raw.oneHot(indices: indices, depth: depthTensor, onValue: onValueTensor, offValue: offValueTensor)
}

@inlinable @inline(__always)
public func ajointRelu<Scalar: BinaryFloatingPoint>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    return Tensor<Scalar>(x.elementsGreater(0))
}

//extension Tensor where Scalar : BinaryFloatingPoint {
//    @inlinable @inline(__always)
//    func adjointConvolved2D(seed: Tensor, originalValue: Tensor, filter: Tensor, strides: (Int32, Int32, Int32, Int32), padding: Padding) -> (Tensor, Tensor) {
//        let dinput: Tensor = Raw.conv2DBackpropInput(inputSizes: shapeTensor, filter: filter, outBackprop: seed, strides: [strides.0, strides.1, strides.2, strides.3], padding: .same)
//        let doutput: Tensor = Raw.conv2DBackpropFilter(self, filterSizes: filter.shapeTensor, outBackprop: seed, strides: [strides.0, strides.1, strides.2, strides.3], padding: padding.raw)
//        return (dinput, doutput)
//        )
//    }
//}
