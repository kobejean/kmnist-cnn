//
//  Types.swift
//  SwiftForTensorFlowTools
//
//  Created by Jean Flaherty on 12/10/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

public typealias TensorFlowInteger = BinaryInteger & TensorFlowScalar

protocol NumericTensor { }
extension Tensor: NumericTensor where Scalar: Numeric {}
