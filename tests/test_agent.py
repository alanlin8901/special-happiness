# test_agent.py

from src.chains.agent_chain import agent  # 請確認 agent 的路徑正確

def main():
    test_input = "1+1=?"
    try:
        response = agent.run(test_input)
        print("Agent 回應:", response)
    except Exception as e:
        import traceback
        print("執行錯誤！")
        traceback.print_exc()

if __name__ == "__main__":
    main()
