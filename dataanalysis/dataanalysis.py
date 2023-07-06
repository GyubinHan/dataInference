import pandas as pd
import matplotlib.pyplot as plt

# 데이터프레임 'data'에 컨테이너 이름, CPU 사용량, 메모리 사용량 등이 포함되어 있다고 가정합니다.
data =pd.read_csv("zipkin-230703.csv",sep='')
# 1. 컨테이너 리소스 사용량 분석
# container_resource_analysis = data.groupby('container').mean()[['cpu_usage', 'memory_usage']]

# 2. API 요청 분석
api_request_analysis = data.groupby('api_name').mean()['duration']

# 3. 시간대별 API 요청 트래픽 분석
time_traffic_analysis = data.groupby(data['timestamp'].dt.hour).count()['api_name']

# 4. 컨테이너 간 리소스 사용량 비교
container_comparison = data.groupby('container').sum()[['cpu_usage', 'memory_usage']]

# 5. API 추적 분석
api_trace_analysis = data.groupby('api_trace_id').mean()['duration']

# 차트 생성
plt.figure(figsize=(10, 6))

# 1. 컨테이너 리소스 사용량 분석
plt.subplot(2, 3, 1)
# container_resource_analysis.plot(kind='bar', ax=plt.gca())
# plt.title('Container Resource Usage')
# plt.xlabel('Container')
# plt.ylabel('Usage')

# 2. API 요청 분석
plt.subplot(2, 3, 2)
api_request_analysis.plot(kind='bar', ax=plt.gca())
plt.title('API Request Duration')
plt.xlabel('API Name')
plt.ylabel('Duration')

# 3. 시간대별 API 요청 트래픽 분석
plt.subplot(2, 3, 3)
time_traffic_analysis.plot(kind='line', ax=plt.gca())
plt.title('Time-based API Traffic')
plt.xlabel('Hour')
plt.ylabel('Request Count')

# 4. 컨테이너 간 리소스 사용량 비교
plt.subplot(2, 3, 4)
container_comparison.plot(kind='bar', ax=plt.gca())
plt.title('Container Resource Comparison')
plt.xlabel('Container')
plt.ylabel('Usage')

# 5. API 추적 분석
plt.subplot(2, 3, 5)
api_trace_analysis.plot(kind='bar', ax=plt.gca())
plt.title('API Trace Analysis')
plt.xlabel('API Trace ID')
plt.ylabel('Duration')

plt.tight_layout()
plt.show()