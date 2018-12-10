//
//  main.swift
//  kmnist
//
//  Created by Jean Flaherty on 12/8/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import Python
import TensorFlow

public let np = Python.import("numpy")

startTrainingOnKMNIST()
