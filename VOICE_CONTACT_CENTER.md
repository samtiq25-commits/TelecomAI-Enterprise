# TelecomAI Voice Contact Center

The voice assistant now supports two modes.

## 1. Customer mode

A customer can speak a complaint such as:

> "My internet has been very slow since this morning."

The system:
1. Transcribes the speech.
2. Classifies the complaint.
3. Checks relevant telecom knowledge.
4. Runs the troubleshooting workflow.
5. Produces a safe customer-facing answer.
6. Converts the answer to speech.

The customer response avoids internal infrastructure information.

## 2. Employee mode

A support agent, NOC engineer, or authorized employee can speak an operational request such as:

> "Tower TWR-1024 has high latency and packet loss."

The system returns:
- complaint/request category
- severity
- diagnosis
- relevant RAG evidence
- recommended operational action
- incident report
- spoken response

## Conversation architecture

```text
Customer / Employee
        ↓
🎙️ Voice
        ↓
Speech-to-Text
        ↓
Audience Detection / Selection
        ↓
Intent + Severity
        ↓
+-----------+-------------+
|           |             |
v           v             v
RAG         ML       Network KPIs
|           |             |
+-----------+-------------+
            ↓
       LangGraph
            ↓
   Audience-specific reply
       ↙           ↘
 Customer          Employee
 response          response
       ↘           ↙
          Text-to-Speech
                ↓
             🔊 Voice
```

## Important boundary

Customer-facing voice responses do not reveal internal infrastructure details. Employee mode is restricted by role and still requires human approval for production network changes.
