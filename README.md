# 4vs4_10000x5000

ver2:


아군이 랜덤으로 스폰된 지역에서 점점 군집을 이루며 앞으로 나아가는 모습을 보임.


랜덤스폰하기 때문에 각 obs에서 자신의 위치와 아군의 위치를 구분하지 못하는 건 아닌가 싶어 스폰지역을 고정해서 학습해볼 필요 있음.


https://github.com/user-attachments/assets/100ab683-73d1-45b8-8828-dfad39e346c2




ver3:

사격행동에 오류가 있어서 그런지 학습이 잘안되는 것 같고, 

현재 obs구조는 자신의 위치와 hp, 아군의 위치와 hp, 상대의 위치와 탐지여부로 구성되어있는데, 위치를 정규화할 때, 맵중심과 크기를 기준으로 정규화하는게 아닌 자신의 위치와 사거리로 정규화 하는 것을 생각하게됨.


그리고 모델의 인코더 수도 현재 obs에 비해 과하다고 생각되서 obs는 그대로 두고 다시 학습 후 판단할 예정.


https://github.com/user-attachments/assets/9cb60344-e1e5-4c43-8252-3df585592fe3


ver4:


흠 뒤로 갔다가 앞으로 돌격하면서 쏘는 행동을 보임. 아마 상대가 직선으로 오는걸 감안해서 대각선으로 간 뒤에 사거리차이를 이용해서 하나하나씩 처치하는게 학습된것같음.


https://github.com/user-attachments/assets/f8d67d67-eb9f-4607-b0e5-389ad9ece29f


https://github.com/user-attachments/assets/2856a618-52d7-4d1c-88b1-4ee70c0b3ac6








