# Planning Time Analysis Feature Documentation

## Overview
Added detailed planning time calculation and statistics functionality to all collaborative lifting demo files for evaluating real-time performance of different models.

## Modified Files
1. `demo_collaborative_lifting_flow_matching.py` - Flow Matching Model
2. `demo_collaborative_lifting_flow_matching_human_condition.py` - Flow Matching Human Condition Model
3. `demo_collaborative_lifting_diffusion_simple.py` - Diffusion Model
4. `demo_collaborative_lifting_diffusion_human_condition.py` - Diffusion Human Condition Model

## Added Features

### 1. Real-time Planning Time Measurement
- Measures model prediction time at each planning step
- Uses high-precision timestamps for millisecond-level accuracy
- Records specific time consumption for each planning operation

### 2. Per-Episode Statistics Output
After each episode, outputs:
- Average planning time
- Maximum planning time
- Minimum planning time

### 3. Final Statistics Summary
After all episodes, outputs:
- 📈 Total planning count
- ⏱️ Average planning time
- 🚀 Fastest planning time
- 🐌 Slowest planning time
- 📊 Standard deviation
- 🎯 Real-time performance evaluation

### 4. Performance Evaluation Standards
- ✅ Excellent: Average planning time < 50ms
- ⚠️ Needs Optimization: Average planning time 50-100ms
- ❌ Poor Performance: Average planning time > 100ms

## Output Examples

### Per-Episode Output
```
Episode 0, fps = 9.8
  📊 Planning Time Stats (ms): Avg=45.23, Max=67.89, Min=32.15
```

### Final Summary
```
============================================================
🎯 Flow Matching Model Planning Time Statistics Summary
============================================================
📈 Total Planning Count: 1250
⏱️  Average Planning Time: 42.35 ms
🚀 Fastest Planning Time: 28.12 ms
🐌 Slowest Planning Time: 89.45 ms
📊 Standard Deviation: 12.34 ms
🎯 Real-time Performance: ✅ Excellent
============================================================
```

## Technical Implementation

### Time Measurement Code
```python
# Record planning time
planning_start = time.time()
robot_action = model.predict_action(observation)
planning_end = time.time()

planning_time = (planning_end - planning_start) * 1000  # Convert to milliseconds
planning_times.append(planning_time)
```

### Statistics Calculation
```python
# Calculate statistical metrics
overall_avg = np.mean(planning_times)
overall_max = np.max(planning_times)
overall_min = np.min(planning_times)
overall_std = np.std(planning_times)
```

## Usage Instructions

1. Run any demo file
2. Observe planning time statistics for each episode
3. View the final performance summary report
4. Judge model real-time performance based on evaluation standards

## Important Notes

- Planning time does not include environment rendering and state update time
- Only measures model inference time to ensure fair comparison
- Recommend performance comparison on the same hardware environment
- Run multiple times and take average for more accurate performance evaluation
