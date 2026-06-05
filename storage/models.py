from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from storage.database import Base
from datetime import datetime, UTC
class EvalRun(Base):
    __tablename__="eval_runs"

    id= Column(Integer,primary_key=True)
    model_name= Column(String)
    run_date = Column(DateTime, default=lambda: datetime.now(UTC))
    total_tasks=Column(Integer)
    avg_f1=Column(Float)
    avg_latency=Column(Float)
    avg_toxicity=Column(Float, nullable=True)
    avg_cost=Column(Float)
    avg_faithfulness=Column(Float, nullable=True)
    avg_relevance=Column(Float, nullable=True)
    results = relationship("EvalResult", back_populates="run")



class EvalResult(Base):
    __tablename__="eval_results"
    
    id=Column(Integer,primary_key=True)
    run_id = Column(Integer, ForeignKey("eval_runs.id"))
    task_id=Column(String)

    category=Column(String)
    prediction=Column(String)
    ground_truth=Column(String)

    exact_match=Column(Float)
    f1=Column(Float)
    latency=Column(Float)
    bert_score = Column(Float, nullable=True)
    cost = Column(Float)
    toxicity = Column(Float, nullable=True)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)

    prompt_tokens=Column(Integer)
    completion_tokens=Column(Integer)

    model_name=Column(String)

    run = relationship("EvalRun", back_populates="results")





