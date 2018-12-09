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
