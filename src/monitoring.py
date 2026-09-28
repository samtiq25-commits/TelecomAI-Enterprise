from .database import recent_incidents

def incident_rows(limit=25):

    rows=[]

    for x in recent_incidents(limit):

        rows.append({

            "id": x.id,
            
            "incident_type": x.incident_type,
        
            "case_status": x.case_status,
            
            "assigned_to": x.assigned_to,
           
            "priority": x.priority,

            "issue": x.issue,

            "tower_id": x.tower_id,

            "thread_id": x.thread_id,

            "severity": x.severity,

            "diagnosis": x.diagnosis,

            "root_cause_confidence": x.root_cause_confidence,

            "recommendation": x.recommendation,

            "created_at": x.created_at,

        })

    return rows