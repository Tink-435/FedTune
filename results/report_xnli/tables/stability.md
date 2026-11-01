| alpha | strategy | runs | final test acc | val acc (last 3 rounds) | oscillation (SD of val acc, rounds 6+) | largest one-round drop |
|---|---|---|---|---|---|---|
| 0.1 | fedavg | 9 | 0.616 | 0.620 | 0.010 | 0.012 |
| 0.1 | diversity | 9 | 0.610 | 0.614 | 0.012 | 0.011 |
| 0.1 | clustered | 9 | 0.606 | 0.617 | 0.013 | 0.013 |
| 0.1 | hybrid | 9 | 0.612 | 0.621 | 0.011 | 0.015 |
| 0.5 | fedavg | 9 | 0.616 | 0.632 | 0.014 | 0.007 |
| 0.5 | diversity | 9 | 0.612 | 0.619 | 0.016 | 0.010 |
| 0.5 | clustered | 9 | 0.616 | 0.627 | 0.015 | 0.009 |
| 0.5 | hybrid | 9 | 0.616 | 0.627 | 0.013 | 0.009 |
| 5 | fedavg | 9 | 0.618 | 0.620 | 0.010 | 0.010 |
| 5 | diversity | 9 | 0.613 | 0.614 | 0.014 | 0.011 |
| 5 | clustered | 9 | 0.616 | 0.615 | 0.011 | 0.014 |
| 5 | hybrid | 9 | 0.617 | 0.618 | 0.010 | 0.015 |

Means over all runs at that alpha. Oscillation is a secondary, post hoc metric.
