# Answer-quality benchmark

Generated from `results/answer_quality_results.json`.

| Variant | Correctness | Faithfulness | Abstention accuracy | Mean latency | Answered |
|---|---:|---:|---:|---:|---:|
| baseline | 70.6% | 94.1% | 90.2% | 2.14s | 51/51 |
| hybrid | 68.6% | 94.1% | 90.2% | 1.86s | 51/51 |
| corrective | 65.9% | 97.6% | 82.4% | 6.61s | 41/51 |
| multiagent | 69.2% | 97.4% | 78.4% | 6.53s | 39/51 |

## Incorrect answered queries

- **baseline:** pol-03, pol-06, pol-08, pol-10, either-02, either-03, either-04, either-05, neither-02, neither-04, faq-01, faq-08, faq-09, faq-14, faq-16
- **hybrid:** pol-03, pol-06, pol-10, either-01, either-02, either-03, either-04, either-07, neither-04, faq-01, faq-08, faq-09, faq-10, faq-11, faq-14, faq-20
- **corrective:** pol-03, pol-04, pol-05, pol-06, either-02, either-03, either-04, either-07, neither-02, neither-04, faq-01, faq-16, faq-20, faq-26
- **multiagent:** pol-03, pol-04, pol-06, either-02, either-04, either-07, neither-02, neither-04, faq-01, faq-08, faq-20, faq-26

## Abstained queries

- **baseline:** none
- **hybrid:** none
- **corrective:** pol-01, pol-10, either-08, neither-01, neither-03, neither-05, faq-06, faq-07, faq-09, faq-12
- **multiagent:** pol-01, pol-10, either-03, either-08, neither-01, neither-03, neither-05, faq-06, faq-07, faq-09, faq-12, faq-16
