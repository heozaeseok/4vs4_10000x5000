# 4vs4_10000x5000

ver2:


아군이 랜덤으로 스폰된 지역에서 점점 군집을 이루며 앞으로 나아가는 모습을 보임.


랜덤스폰하기 때문에 각 obs에서 자신의 위치와 아군의 위치를 구분하지 못하는 건 아닌가 싶어 스폰지역을 고정해서 학습해볼 필요 있음.


https://github.com/user-attachments/assets/100ab683-73d1-45b8-8828-dfad39e346c2




ver3:

사격행동에 오류가 있어서 그런지 학습이 잘안되는 것 같고, 

현재 obs구조는 자신의 위치와 hp, 아군의 위치와 hp, 상대의 위치와 탐지여부로 구성되어있는데, 위치를 정규화할 때, 맵중심과 크기를 기준으로 정규화하는게 아닌 자신의 위치와 사거리로 정규화 하는 것을 생각하게됨.

그리고 모델의 인코더 수도 현재 obs에 비해 과하다고 생각되서 obs는 그대로 두고 다시 학습 후 판단할 예정.

<img width="481" height="355" alt="image" src="https://github.com/user-attachments/assets/d6386906-11ed-417b-92f0-d5e62331be00" />
<img width="474" height="349" alt="image" src="https://github.com/user-attachments/assets/19b2107e-89f5-49aa-b751-20dba1c04047" />
<img width="469" height="341" alt="image" src="https://github.com/user-attachments/assets/b76d27e0-a616-4c47-9741-8ffbbdb6bca5" />
<img width="467" height="346" alt="image" src="https://github.com/user-attachments/assets/8be212ca-a1bb-4d14-9646-ac5ac8679df2" />

https://github.com/user-attachments/assets/9cb60344-e1e5-4c43-8252-3df585592fe3







