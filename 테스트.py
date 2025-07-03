print("룰과 매너를 지켜 즐겁게 듀얼!")
print("듀얼 개시!")
import random
def duel():
    player1 = input("플레이어 1의 이름을 입력하세요: ")
    player2 = input("플레이어 2의 이름을 입력하세요: ")
    
    print(f"{player1}과(와) {player2}의 듀얼이 시작됩니다!")
    
    while True:
        action = input(f"{player1}, 공격할까요? (예/아니오): ")
        if action.lower() == '예':
            damage = random.randint(10, 30)
            print(f"{player1}이(가) {player2}에게 {damage}의 피해를 입혔습니다!")
        else:
            print(f"{player1}은(는) 공격하지 않았습니다.")
        
        action = input(f"{player2}, 공격할까요? (예/아니오): ")
        if action.lower() == '예':
            damage = random.randint(10, 30)
            print(f"{player2}이(가) {player1}에게 {damage}의 피해를 입혔습니다!")
        else:
            print(f"{player2}은(는) 공격하지 않았습니다.")
        
        continue_duel = input("계속 듀얼을 진행하시겠습니까? (예/아니오): ")
        if continue_duel.lower() != '예':
            break
    
    print("듀얼이 종료되었습니다. 감사합니다!") 