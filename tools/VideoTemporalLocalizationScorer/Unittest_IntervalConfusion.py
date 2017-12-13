#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:46:42 2017

@author: tnk12
"""

import unittest
import numpy as np
from intervalcompute import IntervalCompute as IC
from TemporalVideoScoring import VideoScoring

class ConfusionTest(unittest.TestCase):

    """Test case for the confusion measure."""

    def setUp(self):
        """Test initialisation."""
        self.Scorer = VideoScoring()
        pass

    def test_1(self):
        """Basic measure, one interval, no overlap, no collars"""
        global_interval = [0,25]
        ref_intervals = np.array([[5,10]])
        sys_intervals = np.array([[15,20]])
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval)
        self.assertTrue(np.array_equal(np.array([0,1,0,2,0]), cv))
        
    def test_2(self):
        """Basic measure, one interval, partial overlap, no collars"""
        global_interval = [0,25]
        ref_intervals = np.array([[5,15]])
        sys_intervals = np.array([[10,20]])
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval)
        self.assertTrue(np.array_equal(np.array([0,1,3,2,0]), cv))
        
    def test_3(self):
        """Basic measure, one interval, full overlap, system larger than reference, no collars"""
        global_interval = [0,25]
        ref_intervals = np.array([[10,15]])
        sys_intervals = np.array([[5,20]])
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval)
        self.assertTrue(np.array_equal(np.array([0,2,3,2,0]), cv))

    def test_4(self):
        """Basic measure, one interval, full overlap, reference larger than system, no collars"""
        global_interval = [0,25]
        ref_intervals = np.array([[5,20]])
        sys_intervals = np.array([[10,15]])
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval)
        self.assertTrue(np.array_equal(np.array([0,1,3,1,0]), cv))
    
    def test_5(self):
        """Basic measure, ref = 2 intervals, sys = 1, overlap, no collars"""
        global_interval = [0,25]
        ref_intervals = np.array([[5,12], [16, 23]])
        sys_intervals = np.array([[10,18]])
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval)
        self.assertTrue(np.array_equal(np.array([0,1,3,2,3,1,0]), cv))
    
    def test_6(self):
        """Basic measure, ref = 1 intervals, sys = 2, overlap, no collars"""
        global_interval = [0,25]
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[5,12], [16, 23]])
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval)
        self.assertTrue(np.array_equal(np.array([0,2,3,1,3,2,0]), cv))
    
    def test_7(self):
        """test_5 with collars"""
        collar = 1
        global_interval = [0,25]
        ref_intervals = np.array([[5,12], [16, 23]])
        sys_intervals = np.array([[10,18]])
        collars_intervals = IC.compute_collars(ref_intervals, collar, crop_to_range = global_interval)
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval,collars=collars_intervals)
        self.assertTrue(np.array_equal(np.array([0,-1,1,3,-1,2,-1,3,1,-1,0]), cv))
        
    def test_8(self):
        """test_6 with collars"""
        collar = 1
        global_interval = [0,25]
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[5,12], [16, 23]])
        collars_intervals = IC.compute_collars(ref_intervals, collar, crop_to_range = global_interval)
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval,collars=collars_intervals)
        self.assertTrue(np.array_equal(np.array([0,2,-1,3,1,3,-1,2,0]), cv))
    
    def test_9(self):
        """test_8 with a zero collars length, should raise an exception """
        collar = 0
        global_interval = [0,25]
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[5,12], [16, 23]])
        self.assertRaises(AssertionError, IC.compute_collars, ref_intervals, collar, crop_to_range=global_interval)
    
    def test_10(self):
        """test_8 with a negative collars length, should raise an exception """
        collar = -1
        global_interval = [0,25]
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[5,12], [16, 23]])
        self.assertRaises(AssertionError, IC.compute_collars, ref_intervals, collar, crop_to_range=global_interval)
    
    def test_11(self):
        """test_8 with the boundaries inverted for one interval """
        collar = 1
        global_interval = [0,25]
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[5,12], [23, 16]])
        collars_intervals = IC.compute_collars(ref_intervals, collar, crop_to_range = global_interval)
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval,collars=collars_intervals)
        self.assertTrue(np.array_equal(np.array([0,2,-1,3,1,3,-1,2,0]), cv))
    
    def test_12(self):
        """test_8 with intervals shuffled and boundaries inverted  """
        collar = 1
        global_interval = [0,25]
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[23, 16], [5,12]])
        collars_intervals = IC.compute_collars(ref_intervals, collar, crop_to_range = global_interval)
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval,collars=collars_intervals)
        self.assertTrue(np.array_equal(np.array([0,2,-1,3,1,3,-1,2,0]), cv))
    
    def test_13(self):
        """System output has overlap between its own intervals, the union should be performed"""
        collar = 1
        global_interval = [0,25]
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[5,16], [12, 20]])
        collars_intervals = IC.compute_collars(ref_intervals, collar, crop_to_range = global_interval)
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval,collars=collars_intervals)
        self.assertTrue(np.array_equal(np.array([0,2,-1,3,-1,2,0]), cv))
        
    def test_14(self):
        """System output has some interval of zero length, we delete them"""
        collar = 1
        global_interval = [0,25]
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[5,12], [14,14], [17, 20]])
        collars_intervals = IC.compute_collars(ref_intervals, collar, crop_to_range = global_interval)
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval,collars=collars_intervals)
        self.assertTrue(np.array_equal(np.array([ 0,2,-1,3,1,-1,2,0]), cv))
        
    def test_15(self):
        """Adding two system no-score zones"""
        collar = 1
        global_interval = [0,25]
        SNS = np.array([[4,9], [13,16]])
        ref_intervals = np.array([[10,18]])
        sys_intervals = np.array([[5,12], [17, 20]])
        collars_intervals = IC.compute_collars(ref_intervals, collar, crop_to_range = global_interval)
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval,collars=collars_intervals, SNS=SNS)
        self.assertTrue(np.array_equal(np.array([0,-2,-1,3,1,-2,1,-1,2,0]), cv))
        
    def test_16(self):
        """Empty ref intervals case scenario. No collars or SNS"""
        global_interval = [0,25]
        SNS = np.array([[4,9], [13,16]])
        ref_intervals = np.array([[]])
        sys_intervals = np.array([[5,12], [17, 20]])
        cv,all_intervals,_ = self.Scorer.compute_confusion_map(ref_intervals, sys_intervals, global_interval,SNS=SNS)
        self.assertTrue(np.array_equal(np.array([0,-2,2,0,-2,0,2,0]), cv))

        
if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False,verbosity=1)
    unittest.main(verbosity=1)