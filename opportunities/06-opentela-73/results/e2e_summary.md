| binary | scenario | history | restart delay (s) | runs | routable after restart, s: median (min-max) | not routable within timeout | worker's own view | POST workaround, s | guard fired, s |
|---|---|---|---|---|---|---|---|---|---|
| base | pub | 0 | 0 | 3 | n/a | 3/3 | ready/1 svc | 15.5 (14.8-15.5) | n/a |
| fixed | pub | 0 | 0 | 3 | 15.5 (15.3-15.6) | 0/3 | ready/1 svc | n/a | 5.2 (3.3-12.5) |
| rejoin | pub | 0 | 0 | 3 | 8.4 (1.3-11.1) | 0/3 | ready/1 svc | n/a | 10.7 (3.1-13.0) |
| base | pub | 0 | 20 | 3 | n/a | 3/3 | ready/1 svc | 20.0 (19.9-20.1) | n/a |
| fixed | pub | 0 | 20 | 3 | 19.9 (19.9-20.0) | 0/3 | ready/1 svc | n/a | 9.7 (7.7-11.0) |
| rejoin | pub | 0 | 20 | 3 | 8.6 (2.8-9.9) | 0/3 | ready/1 svc | n/a | 10.6 (4.8-11.9) |
| base | pub | 0 | 70 | 3 | n/a | 3/3 | ready/1 svc | 5.2 (5.2-5.2) | n/a |
| fixed | pub | 0 | 70 | 3 | 20.0 (19.9-20.1) | 0/3 | ready/1 svc | n/a | 8.9 (7.4-9.4) |
| rejoin | pub | 0 | 70 | 3 | 6.5 (1.8-9.6) | 0/3 | ready/1 svc | n/a | 8.5 (3.7-11.6) |
| base | relay | 20000 | 0 | 2 | 19.9 (19.8-19.9) | 0/2 | ready/1 svc | n/a | n/a |
| fixed | relay | 20000 | 0 | 3 | 19.8 (19.8-19.9) | 0/3 | ready/1 svc | n/a | 5.6 (5.4-6.6) |
