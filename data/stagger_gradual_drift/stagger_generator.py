#!/usr/bin/env python3

from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.stagger_generator import STAGGERGenerator
import pandas as pd
import numpy as np

n_samples = 1000000
for seed in range(30, 100):
    base_stream_generator = STAGGERGenerator(classification_function=1,
                                             random_state=1,
                                             balance_classes=False)

    drift_stream_generator = STAGGERGenerator(classification_function=2,
                                              random_state=2,
                                              balance_classes=False)

    generator = ConceptDriftStream(stream=base_stream_generator,
                                   drift_stream=drift_stream_generator,
                                   random_state=seed,
                                   alpha=0,
                                   position=0,
                                   width=1000)

    generator.prepare_for_use()

    X_raw, y = generator.next_sample(n_samples)

    X = []
    for i in range(0, n_samples):
        one_hot_encoded_row = []
        for value in X_raw[i]:
            for j in range(0, 3):
                one_hot_encoded_row.append(1 if j == value else 0)
        X.append(one_hot_encoded_row)

    df = pd.DataFrame(np.hstack((X, np.array([y]).T)))
    df = df.astype(int)

    file_name = "{:02d}.csv".format(seed);
    df.to_csv(file_name, index=False)
