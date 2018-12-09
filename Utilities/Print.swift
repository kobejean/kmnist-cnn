//
//  Print.swift
//  KMNIST
//
//  Created by Jean Flaherty on 12/9/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import Foundation

func printDividerLine() {
    print("-----------------------------------------------------------------")
}

func printTrainingLoss(epochNumber: Int, batchNumber: Int, H_total: Float) {
    print(String(format: "Epoch: %03d Batch: %04d Training Loss: %.5f Accuracy: %.5f", epochNumber, batchNumber, H_total, exp(-H_total)))
}

func printValidationLoss(epochNumber: Int, batchNumber: Int, H_test: Float) {
    print(String(format: "Epoch: %03d Batch: %04d Validation Loss: %.5f Accuracy: %.5f", epochNumber, batchNumber, H_test, exp(-H_test)))
}
