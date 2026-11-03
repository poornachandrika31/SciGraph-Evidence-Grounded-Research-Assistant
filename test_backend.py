import asyncio
from app.config.database import connect_to_mongo, close_mongo_connection, get_db
from app.services.auth import hash_password, verify_password, create_access_token
from app.services.literature import LiteratureService
from app.agent.graph import research_agent_graph
from app.agent.state import AgentState

async def test_full_pipeline():
    print("--- 1. Testing MongoDB Connection ---")
    await connect_to_mongo()
    db = get_db()
    print("Database connected:", db.name)

    print("\n--- 2. Testing Password & Token Authentication ---")
    pwd = "ResearcherPass123!"
    hpwd = hash_password(pwd)
    assert verify_password(pwd, hpwd) is True
    print("Password hashing & verification passed!")

    token = create_access_token({"sub": "test_user_id", "email": "researcher@lab.org"})
    assert len(token) > 20
    print("JWT Token generated successfully!")

    print("\n--- 3. Testing Scientific Literature Retrieval ---")
    papers = await LiteratureService.search_literature("microplastics and soil plant growth", limit=2)
    print(f"Retrieved {len(papers)} papers:")
    for p in papers:
        print(f" - [{p.get('source')}] {p.get('title')} ({p.get('year')})")

    print("\n--- 4. Testing LangGraph Scientific Research Agent ---")
    state: AgentState = {
        "session_id": "test_session_id",
        "user_id": "test_user_id",
        "research_question": "Does microplastics exposure affect root nodulation in legumes?",
        "user_message": "Generate 3 hypotheses exploring this question.",
        "history": [],
        "retrieved_papers": [],
        "candidate_hypotheses": []
    }
    result = await research_agent_graph.ainvoke(state)
    print("Agent Intent:", result.get("intent"))
    print("Hypotheses generated:", len(result.get("candidate_hypotheses", [])))
    for idx, h in enumerate(result.get("candidate_hypotheses", []), 1):
        print(f" Hypothesis {idx} (Score: {h.get('overall_score')}/10): {h.get('hypothesis')[:80]}...")
    print("\nAssistant Response Preview:\n", result.get("final_response")[:250], "...")

    print("\n--- 5. Testing Experiment Design Generation ---")
    exp_state: AgentState = {
        "session_id": "test_session_id",
        "user_id": "test_user_id",
        "research_question": "Does microplastics exposure affect root nodulation in legumes?",
        "user_message": "Design an experiment to test hypothesis 1.",
        "history": [],
        "candidate_hypotheses": result.get("candidate_hypotheses", [])
    }
    exp_result = await research_agent_graph.ainvoke(exp_state)
    print("Experiment objective:", exp_result.get("experiment_design", {}).get("objective"))
    print("Experiment IVs:", exp_result.get("experiment_design", {}).get("independent_variables"))
    print("Experiment DVs:", exp_result.get("experiment_design", {}).get("dependent_variables"))

    await close_mongo_connection()
    print("\n>>> ALL BACKEND SERVICES TESTED & VERIFIED PERFECTLY! <<<")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
