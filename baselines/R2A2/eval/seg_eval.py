#!/usr/bin/python2.7
# code from https://github.com/yabufarha/ms-tcn/blob/master/eval.py (MIT License)
# yabufarha adapted it from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py (MIT License)

import os
import numpy as np
import argparse


class MetricsSegments(object):
    """Class to compute segment level metrics.
    
    """
    def __init__(self):
        pass

    def levenstein(self, p, y, norm=False):
        """Calculates the Levenstein distance between two sequences."""
        m_row = len(p)
        n_col = len(y)
        D = np.zeros([m_row+1, n_col+1], float)
        for i in range(m_row+1):
            D[i, 0] = i
        for i in range(n_col+1):
            D[0, i] = i

        for j in range(1, n_col+1):
            for i in range(1, m_row+1):
                if y[j-1] == p[i-1]:
                    D[i, j] = D[i-1, j-1]
                else:
                    D[i, j] = min(D[i-1, j] + 1,
                                D[i, j-1] + 1,
                                D[i-1, j-1] + 1)
        if norm:
            score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
        else:
            score = D[-1, -1]

        return score
    
    def edit_score(self, recognized, ground_truth, norm=True, bg_class=-1):
        """Calculates the edit score between two sequences."""
        P, _, _ = self.get_labels_start_end_time(recognized, bg_class)
        Y, _, _ = self.get_labels_start_end_time(ground_truth, bg_class)
        return self.levenstein(P, Y, norm)


    def f_score(self, recognized, ground_truth, overlap, bg_class=-1):
        """Calculates the F-score between two sequences."""
        p_label, p_start, p_end = self.get_labels_start_end_time(recognized, bg_class)
        y_label, y_start, y_end = self.get_labels_start_end_time(ground_truth, bg_class)

        tp = 0
        fp = 0

        hits = np.zeros(len(y_label))

        for j in range(len(p_label)):
            intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
            union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
            IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
            # Get the best scoring segment
            idx = np.array(IoU).argmax()

            if IoU[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1
        fn = len(y_label) - sum(hits)
        return float(tp), float(fp), float(fn)


    def accuracy(self, recog_content, gt_content):
        correct = sum([1 for r, g in zip(recog_content, gt_content) if r == g])
        total = len(recog_content)
        return correct / total


    def get_labels_start_end_time(self, frame_wise_labels, bg_class=-1):
        labels = []
        starts = []
        ends = []
        last_label = frame_wise_labels[0]
        if frame_wise_labels[0] != bg_class:
            labels.append(frame_wise_labels[0])
            starts.append(0)
        for i in range(len(frame_wise_labels)):
            if frame_wise_labels[i] != last_label:
                if frame_wise_labels[i] != bg_class:
                    labels.append(frame_wise_labels[i])
                    starts.append(i)
                if last_label != bg_class:
                    ends.append(i)
                last_label = frame_wise_labels[i]
        if last_label != bg_class:
            ends.append(i)
        return labels, starts, ends

    def get_metrics(self, pred_vids, gt_vids, overlap=[.1, .25, .5]):
        """Computes the metrics for a list of videos.

        Args:
            gt (list of lists): list of ground truth labels
            pred (list of lists): list of predicted labels
            overlap (list): list of overlap thresholds

        Returns:
            acc (float): accuracy
            edit (float): edit score
            f1s (list): list of f1 scores
        """
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct = 0
        total = 0
        edit = 0

        count_vids = 0

        for vid_i in range(len(gt_vids)):
            recog_content = pred_vids[vid_i]
            gt_content = gt_vids[vid_i]

            if len(recog_content) < 1:
                print('No prediction for video %d' % vid_i)
                continue
            
            for i in range(len(recog_content)):
                total += 1
                if recog_content[i] == gt_content[i]:
                    correct += 1

            edit += self.edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = self.f_score(recog_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1
            
            count_vids += 1

        # print("Acc (frame-level)(not-relaxed): %.4f" % (100*float(correct)/total))
        # print('Edit: %.4f' % ((1.0*edit)/len(gt_vids)))
        acc = (100*float(correct)/total)
        edit = ((1.0*edit)/count_vids)
        f1s = []
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])

            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100
            # print('F1@%0.2f: %.4f' % (overlap[s], f1))
            f1s.append(f1)
        
        return acc, edit, f1s






