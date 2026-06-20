# Research Background

## Problem

The project classifies Reddit mental-health-related posts into six classes:

- ADHD
- Anxiety
- Bipolar
- Depression
- PTSD
- None

The task is supervised multi-class text classification. Each post is represented
using both neural semantic representations and interpretable psycholinguistic
features.

## Research Motivation

Mental health text classification benefits from two complementary signals:

1. Semantic signal from a domain-adapted language model.
2. Interpretable style, affect, and discourse signals from handcrafted features.

The fusion setting tests whether interpretable features add value beyond a
strong MentalRoBERTa baseline and whether branch weights can expose which
feature families matter for each class.

## Main Research Questions

1. Does combining psycholinguistic features with MentalRoBERTa improve macro-F1?
2. Are gains consistent across random seeds?
3. Which classes rely more on semantic, affective, or handcrafted branches?
4. Can a fusion model provide interpretable branch-level evidence without
   sacrificing performance?

## Ethical Framing

This is a classification research project, not a clinical diagnostic tool.
Outputs should be interpreted as dataset-level model behavior. They should not
be used to infer an individual user's mental health condition in practice.

