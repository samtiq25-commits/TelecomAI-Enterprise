import random
from datetime import datetime,timedelta
import numpy as np,pandas as pd
from .config import (
    NETWORK_CSV,
    EQUIPMENT_CSV,
    CUSTOMER_CSV,
    NETWORK_EVENTS_CSV,
)
CATEGORIES=["Internet Speed","Call Drops","No Network","SMS Failure","SIM Problems","Billing","Package Issues","Network Outage"]
TEXTS={
"Internet Speed":["Internet is extremely slow since yesterday","Mobile data speed has dropped badly","4G is very slow in my area"],
"Call Drops":["My calls keep dropping","Calls disconnect after a few minutes"],
"No Network":["There is no network signal","My phone shows no service"],
"SMS Failure":["SMS messages are not going through","I cannot send text messages"],
"SIM Problems":["My SIM stopped working","SIM is not being detected"],
"Billing":["I was charged incorrectly","My bill is higher than expected"],
"Package Issues":["My internet package is not active","The package benefits are missing"],
"Network Outage":["The whole area seems to have an outage","Network service is unavailable in my area"]}
def generate_network_events(tower_ids, n_events=150):
    events = []

    event_types = [
        "BACKHAUL",
        "MAINTENANCE",
        "CONFIGURATION"
    ]

    start = datetime.now() - timedelta(hours=300)

    for i in range(n_events):
        tower = random.choice(tower_ids)
        timestamp = start + timedelta(
            hours=random.randint(0, 300)
        )

        event_type = random.choice(event_types)

        if event_type == "BACKHAUL":
            utilization = float(
                np.clip(
                    np.random.normal(65, 15),
                    20,
                    100
                )
            )

            severity = (
                "HIGH"
                if utilization >= 85
                else "NORMAL"
            )

            description = (
                f"Backhaul utilization recorded at "
                f"{utilization:.1f}%."
            )

            events.append([
                timestamp,
                tower,
                event_type,
                severity,
                utilization,
                "",
                description
            ])

        elif event_type == "MAINTENANCE":
            maintenance_type = random.choice([
                "Preventive maintenance",
                "Equipment inspection",
                "Link maintenance",
                "Routine service"
            ])

            severity = random.choice([
                "NORMAL",
                "NORMAL",
                "LOW"
            ])

            description = (
                f"{maintenance_type} performed."
            )

            events.append([
                timestamp,
                tower,
                event_type,
                severity,
                "",
                maintenance_type,
                description
            ])

        else:
            configuration_type = random.choice([
                "Routing update",
                "Radio configuration update",
                "QoS configuration update",
                "Network parameter update"
            ])

            severity = random.choice([
                "NORMAL",
                "LOW",
                "MEDIUM"
            ])

            description = (
                f"{configuration_type} recorded."
            )

            events.append([
                timestamp,
                tower,
                event_type,
                severity,
                "",
                configuration_type,
                description
            ])

    return pd.DataFrame(
        events,
        columns=[
            "timestamp",
            "tower_id",
            "event_type",
            "severity",
            "backhaul_utilization",
            "event_name",
            "description"
        ]
    )
def generate_all(n_towers=30,periods=300,n_equipment=250,n_complaints=600):
    random.seed(42); np.random.seed(42)
    towers=[f"TWR-{1000+i}" for i in range(n_towers)]
    rows=[]; start=datetime.now()-timedelta(hours=periods)
    for i in range(periods):
        ts=start+timedelta(hours=i)
        for tower in towers:
            traffic=max(5,np.random.normal(65,18)); latency=max(10,np.random.normal(70,20))
            loss=max(0,np.random.normal(1.8,.9)); signal=np.random.normal(-88,10)
            cpu=min(100,max(5,np.random.normal(55,15))); mem=min(100,max(5,np.random.normal(58,13)))
            drops=max(0,np.random.normal(1.5,.8))
            if np.random.random()<.025:
                latency*=np.random.uniform(2,4); loss*=np.random.uniform(2,5); cpu=min(100,cpu+np.random.uniform(25,40))
            rows.append([ts,tower,traffic,latency,loss,signal,cpu,mem,drops])
    network=pd.DataFrame(rows,columns=["timestamp","tower_id","network_traffic","latency","packet_loss","signal_strength","cpu_usage","memory_usage","call_drop_rate"])
   
    tower_ids = network["tower_id"].drop_duplicates().tolist()

    # Generate network operational events
    network_events = generate_network_events(tower_ids)

    eq = []

    for i in range(n_equipment):
        tower = tower_ids[i % len(tower_ids)]

        temp=np.random.normal(55,10)
        uptime=np.random.uniform(100,5000)
        cpu=np.random.normal(60,18)
        mem=np.random.normal(60,15)
        errors=np.random.poisson(5)
        maint=np.random.randint(0,8)

        failure=int(
            temp>75
            or cpu>90
            or errors>12
            or (uptime>4500 and maint<2)
        )

        eq.append([
            f"EQ-{5000+i}",
            tower,
            temp,
            uptime,
            cpu,
            mem,
            errors,
            maint,
            failure
        ])

    equipment=pd.DataFrame(
        eq,
        columns=[
            "equipment_id",
            "tower_id",
            "temperature",
            "uptime",
            "cpu_usage",
            "memory_usage",
            "error_count",
            "maintenance_count",
            "failure"
        ]
    )
    cr=[]
    locs=[f"Area-{i+1}" for i in range(n_towers)]
    services=["Prepaid", "Postpaid", "Broadband", "Enterprise"]
    for i in range(n_complaints):
        cat=random.choice(CATEGORIES); sev=random.choices(["LOW","MEDIUM","HIGH","CRITICAL"],[35,40,20,5])[0]
        cr.append([f"CUST-{90000+i}",random.choice(TEXTS[cat]),random.choice(locs),random.choice(services),cat,sev,random.randint(5,240)])
    customers=pd.DataFrame(cr,columns=["customer_id","complaint","location","service_type","complaint_category","severity","resolution_time"])
    network.to_csv(NETWORK_CSV, index=False)
    equipment.to_csv(EQUIPMENT_CSV, index=False)
    customers.to_csv(CUSTOMER_CSV, index=False)
    network_events.to_csv(NETWORK_EVENTS_CSV, index=False)

    return network, equipment, customers
def load_data():
    required_files = [
        NETWORK_CSV,
        EQUIPMENT_CSV,
        CUSTOMER_CSV,
        NETWORK_EVENTS_CSV,
    ]

    if not all(p.exists() for p in required_files):
        return generate_all()

    return (
        pd.read_csv(
            NETWORK_CSV,
            parse_dates=["timestamp"]
        ),
        pd.read_csv(EQUIPMENT_CSV),
        pd.read_csv(CUSTOMER_CSV),
    )