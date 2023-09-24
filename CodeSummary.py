# 중복된 항목 수 알아보기
print("중복된 항목 수 :", len(data[data.duplicated()])) 

# 중복된 항목 확인
print(data[data.duplicated(keep = False)].sort_values(by=list(data.columns)).head())

# 중복된 항목 제거 및 중복 값 한 개는 남겨둠
data.drop_duplicates(inplace=True, keep='first', ignore_index = True)