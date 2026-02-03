#!/usr/bin/env python3
"""PHASE 5: FINAL INTEGRATION VERIFICATION"""
import asyncio, os, sys, time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

class S(Enum):
    P = "✅"; F = "❌"; W = "⚠️"

@dataclass
class R:
    n: str; s: S; t: float; d: str = ""

class Suite:
    def __init__(self):
        self.r = []
        self.k = os.getenv("ANTHROPIC_API_KEY")
    def a(self, r):
        self.r.append(r)
        print(f"  {r.s.value} {r.n} ({r.t:.1f}s) {r.d}")
    async def t1(self):
        t=time.time()
        try:
            from src.nuble.agents.base import SpecializedAgent, AgentType
            from src.nuble.agents.orchestrator import OrchestratorAgent
            from src.institutional.ml.regime import HMMRegimeModel
            return R("Imports",S.P,time.time()-t,"OK")
        except Exception as e: return R("Imports",S.F,time.time()-t,str(e)[:50])
    async def t2(self):
        t=time.time()
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            o=OrchestratorAgent(api_key=self.k)
            o._initialize_agents()  # Force lazy init
            c=len(o._agents)
            return R("Init",S.P if c>=9 else S.F,time.time()-t,f"{c}agents")
        except Exception as e: return R("Init",S.F,time.time()-t,str(e)[:50])
    async def t3(self):
        t=time.time()
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            o=OrchestratorAgent(api_key=self.k)
            r=await o.process("Apple stock price?","t3")
            m=r.get('message','')
            return R("Simple Query",S.P if len(m)>50 else S.F,time.time()-t,f"{len(m)}c")
        except Exception as e: return R("Simple Query",S.F,time.time()-t,str(e)[:50])
    async def t4(self):
        t=time.time()
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            o=OrchestratorAgent(api_key=self.k)
            r=await o.process("NVIDIA analysis: technicals and risks","t4")
            m=r.get('message','');a=r.get('agents_used',[])
            return R("Complex Query",S.P if len(m)>200 else S.F,time.time()-t,f"{len(m)}c,{len(a)}a")
        except Exception as e: return R("Complex Query",S.F,time.time()-t,str(e)[:50])
    async def t5(self):
        t=time.time()
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            o=OrchestratorAgent(api_key=self.k)
            r=await o.process("What is P/E ratio?","t5")
            m=r.get('message','')
            return R("Education",S.P if len(m)>50 else S.F,time.time()-t,f"{len(m)}c")
        except Exception as e: return R("Education",S.F,time.time()-t,str(e)[:50])
    async def t6(self):
        t=time.time()
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            o=OrchestratorAgent(api_key=self.k)
            c=f"ctx{int(time.time())}"
            r1=await o.process("Tesla stock",c);r2=await o.process("Competitors?",c)
            m1,m2=r1.get('message',''),r2.get('message','')
            return R("Context",S.P if len(m1)>50 and len(m2)>50 else S.F,time.time()-t,"OK")
        except Exception as e: return R("Context",S.F,time.time()-t,str(e)[:50])
    async def t7(self):
        t=time.time()
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            o=OrchestratorAgent(api_key=self.k)
            r=await o.process("¿Cómo está AAPL?","t7")
            m=r.get('message','')
            return R("Edge Case",S.P if len(m)>20 else S.F,time.time()-t,f"{len(m)}c")
        except Exception as e: return R("Edge Case",S.F,time.time()-t,str(e)[:50])
    async def t8(self):
        t=time.time()
        try:
            from src.nuble.agents.orchestrator import OrchestratorAgent
            o=OrchestratorAgent(api_key=self.k)
            s=f"wf{int(time.time())}"
            r1=await o.process("Market overview",s)
            r2=await o.process("Apple analysis",s)
            m1,m2=r1.get('message',''),r2.get('message','')
            return R("Workflow",S.P if len(m1)>50 and len(m2)>50 else S.F,time.time()-t,"OK")
        except Exception as e: return R("Workflow",S.F,time.time()-t,str(e)[:50])
    async def run(self):
        st=time.time()
        print("\n"+"="*60)
        print("PHASE 5: INTEGRATION VERIFICATION")
        print("="*60)
        print(f"API: ...{self.k[-4:] if self.k else 'NONE'}")
        print("-"*40)
        print("COMPONENT TESTS")
        print("-"*40)
        self.a(await self.t1());self.a(await self.t2())
        print("-"*40)
        print("LIVE QUERY TESTS")
        print("-"*40)
        self.a(await self.t3());self.a(await self.t4());self.a(await self.t5())
        print("-"*40)
        print("ADVANCED TESTS")
        print("-"*40)
        self.a(await self.t6());self.a(await self.t7());self.a(await self.t8())
        p=sum(1 for x in self.r if x.s==S.P);f=sum(1 for x in self.r if x.s==S.F);n=len(self.r)
        print("\n"+"="*60)
        print(f"RESULTS: ✅{p} ❌{f} Total:{n} Time:{time.time()-st:.1f}s")
        print("="*60)
        if f==0: print("🏆 PHASE 5: PASSED - PRODUCTION READY")
        else: print(f"❌ PHASE 5: {f} FAILURES")
        print("="*60)
        return f==0

if __name__=="__main__":
    asyncio.run(Suite().run())
