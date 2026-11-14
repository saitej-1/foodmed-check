"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Activity, Shield, AlertTriangle, CheckCircle, Brain, 
  RefreshCw, BarChart2, Search, Cpu, Database, ClipboardList, 
  Info, HelpCircle, ArrowRight, UserCheck, Star, Sparkles
} from "lucide-react";

// API endpoints
const API_HOST = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export default function Home() {
  const [activeTab, setActiveTab] = useState<"checker" | "sandbox" | "history" | "admin">("checker");

  // Checker State
  const [drugInput, setDrugInput] = useState("");
  const [foodInput, setFoodInput] = useState("");
  const [drugSmiles, setDrugSmiles] = useState("");
  const [foodSmiles, setFoodSmiles] = useState("");
  
  const [drugSuggestions, setDrugSuggestions] = useState<any[]>([]);
  const [foodSuggestions, setFoodSuggestions] = useState<any[]>([]);
  
  const [showDrugDropdown, setShowDrugDropdown] = useState(false);
  const [showFoodDropdown, setShowFoodDropdown] = useState(false);
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState<any>(null);
  
  // SHAP Explainability State
  const [isExplaining, setIsExplaining] = useState(false);
  const [shapResult, setShapResult] = useState<any>(null);

  // Sandbox State
  const [sandboxSmiles, setSandboxSmiles] = useState("CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"); // Warfarin default
  const [sandboxMoleculeName, setSandboxMoleculeName] = useState("Warfarin");
  const [sandboxFormula, setSandboxFormula] = useState("C19H16O4");
  const [sandboxWeight, setSandboxWeight] = useState(308.33);

  // History State
  const [historyList, setHistoryList] = useState<any[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  
  // User Feedback State
  const [userRating, setUserRating] = useState(5);
  const [userComment, setUserComment] = useState("");
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  // Admin Dashboard State
  const [platformStats, setPlatformStats] = useState<any>(null);
  const [modelMetadata, setModelMetadata] = useState<any>(null);
  const [retrainLogs, setRetrainLogs] = useState("");
  const [retrainStatus, setRetrainStatus] = useState("idle");
  const [isRetraining, setIsRetraining] = useState(false);

  // Sandbox molecular layout canvas ref
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);

  // Fetch suggestions
  useEffect(() => {
    if (drugInput.trim().length >= 1) {
      const delay = setTimeout(() => {
        fetch(`${API_HOST}/api/search/drugs?q=${encodeURIComponent(drugInput)}`)
          .then(res => res.json())
          .then(data => {
            setDrugSuggestions(data);
            setShowDrugDropdown(true);
          })
          .catch(err => console.error("Error fetching drug suggestions:", err));
      }, 200);
      return () => clearTimeout(delay);
    } else {
      setDrugSuggestions([]);
      setShowDrugDropdown(false);
    }
  }, [drugInput]);

  useEffect(() => {
    if (foodInput.trim().length >= 1) {
      const delay = setTimeout(() => {
        fetch(`${API_HOST}/api/search/foods?q=${encodeURIComponent(foodInput)}`)
          .then(res => res.json())
          .then(data => {
            setFoodSuggestions(data);
            setShowFoodDropdown(true);
          })
          .catch(err => console.error("Error fetching food suggestions:", err));
      }, 200);
      return () => clearTimeout(delay);
    } else {
      setFoodSuggestions([]);
      setShowFoodDropdown(false);
    }
  }, [foodInput]);

  // Load platform stats & history when tabs change
  useEffect(() => {
    if (activeTab === "admin") {
      fetchAdminStats();
    } else if (activeTab === "history") {
      fetchHistory();
    }
  }, [activeTab]);

  const fetchAdminStats = () => {
    fetch(`${API_HOST}/api/admin/stats`)
      .then(res => res.json())
      .then(data => {
        setPlatformStats(data.statistics);
        if (data.recent_predictions) {
          setHistoryList(data.recent_predictions);
        }
      })
      .catch(err => console.error("Error fetching admin stats:", err));

    fetch(`${API_HOST}/api/admin/model-info`)
      .then(res => res.json())
      .then(data => setModelMetadata(data))
      .catch(err => console.error("Error fetching model info:", err));
      
    // Poll retraining log status
    fetch(`${API_HOST}/api/admin/retrain/status`)
      .then(res => res.json())
      .then(data => {
        setRetrainStatus(data.status);
        setRetrainLogs(data.log);
      })
      .catch(err => console.error(err));
  };

  const fetchHistory = () => {
    setIsLoadingHistory(true);
    fetch(`${API_HOST}/api/admin/stats`)
      .then(res => res.json())
      .then(data => {
        if (data.recent_predictions) {
          setHistoryList(data.recent_predictions);
        }
      })
      .catch(err => console.error(err))
      .finally(() => setIsLoadingHistory(false));
  };

  const runPrediction = async () => {
    if (!drugSmiles && !drugInput) return;
    if (!foodSmiles && !foodInput) return;

    setIsAnalyzing(true);
    setPredictionResult(null);
    setShapResult(null);
    setFeedbackSubmitted(false);

    try {
      const res = await fetch(`${API_HOST}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          drug_name: drugInput,
          food_name: foodInput,
          drug_smiles: drugSmiles || undefined,
          food_smiles: foodSmiles || undefined
        })
      });
      
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction request failed.");
      }

      const data = await res.json();
      setPredictionResult(data);
      
      // Instantly trigger SHAP explanations
      runShapExplain(data.drug.smiles, data.food.smiles);
    } catch (e: any) {
      alert(`Error running prediction: ${e.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const runShapExplain = async (dSmiles: string, fSmiles: string) => {
    setIsExplaining(true);
    try {
      const res = await fetch(`${API_HOST}/api/explain?drug_smiles=${encodeURIComponent(dSmiles)}&food_smiles=${encodeURIComponent(fSmiles)}`);
      const data = await res.json();
      setShapResult(data);
    } catch (e) {
      console.error("SHAP explain failed", e);
    } finally {
      setIsExplaining(false);
    }
  };

  const runRetrain = async () => {
    setIsRetraining(true);
    try {
      await fetch(`${API_HOST}/api/admin/retrain`, { method: "POST" });
      setRetrainStatus("running");
      setRetrainLogs("Retraining triggered... check status log below in a few seconds.");
      // Poll every 3 seconds
      const poll = setInterval(async () => {
        const res = await fetch(`${API_HOST}/api/admin/retrain/status`);
        const statusData = await res.json();
        setRetrainStatus(statusData.status);
        setRetrainLogs(statusData.log);
        if (statusData.status === "completed" || statusData.status === "failed" || statusData.status === "idle") {
          clearInterval(poll);
          setIsRetraining(false);
          fetchAdminStats(); // Refresh active metadata
        }
      }, 3000);
    } catch (e) {
      console.error(e);
      setIsRetraining(false);
    }
  };

  const submitFeedback = async (predId: number) => {
    try {
      const res = await fetch(`${API_HOST}/api/predict/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction_id: predId,
          user_rating: userRating,
          user_comment: userComment
        })
      });
      if (res.ok) {
        setFeedbackSubmitted(true);
        setUserComment("");
      }
    } catch (e) {
      console.error(e);
    }
  };

  // ----------------------------------------------------
  // Interactive 2D/3D Molecule layout inside Sandbox Canvas
  // ----------------------------------------------------
  useEffect(() => {
    if (activeTab !== "sandbox" || !canvasRef.current) return;

    // Simple SMILES-like layout simulator
    const atoms: any[] = [];
    const bonds: any[] = [];
    
    // Parse characters of smiles to create a visual graph
    let activeIdx = -1;
    const branchStack: number[] = [];
    const ringMap: { [key: string]: number } = {};
    
    let curIdx = 0;
    while (curIdx < sandboxSmiles.length) {
      const char = sandboxSmiles[curIdx];
      if (char === '(') {
        branchStack.push(activeIdx);
        curIdx++;
        continue;
      }
      if (char === ')') {
        activeIdx = branchStack.pop() ?? -1;
        curIdx++;
        continue;
      }
      if (char === '=' || char === '#' || char === '-') {
        curIdx++;
        continue;
      }
      if (char >= '0' && char <= '9') {
        if (ringMap[char] !== undefined) {
          bonds.push({ source: activeIdx, target: ringMap[char] });
          ringMap[char] = -1;
        } else {
          ringMap[char] = activeIdx;
        }
        curIdx++;
        continue;
      }
      
      // Parse elements
      let element = "";
      if (curIdx < sandboxSmiles.length - 1 && ["CL", "CA", "BR", "NA", "MG"].includes(sandboxSmiles.substr(curIdx, 2).toUpperCase())) {
        element = sandboxSmiles.substr(curIdx, 2).toUpperCase();
        curIdx += 2;
      } else if (["C", "H", "O", "N", "F", "P", "S", "K", "I"].includes(char.toUpperCase())) {
        element = char.toUpperCase();
        curIdx++;
      } else {
        curIdx++;
        continue;
      }
      
      const newAtom = {
        id: atoms.length,
        element,
        x: (Math.random() - 0.5) * 100,
        y: (Math.random() - 0.5) * 100,
        z: (Math.random() - 0.5) * 100
      };
      
      atoms.push(newAtom);
      if (activeIdx !== -1) {
        bonds.push({ source: activeIdx, target: newAtom.id });
      }
      activeIdx = newAtom.id;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = canvas.width = canvas.parentElement?.clientWidth || 500;
    let height = canvas.height = 400;

    let angleX = 0.005;
    let angleY = 0.005;

    // Physics spring simulation
    const runSimulation = () => {
      // Rotate atoms in 3D
      atoms.forEach(atom => {
        // Rotate X
        const y1 = atom.y * Math.cos(angleX) - atom.z * Math.sin(angleX);
        const z1 = atom.y * Math.sin(angleX) + atom.z * Math.cos(angleX);
        // Rotate Y
        const x2 = atom.x * Math.cos(angleY) - z1 * Math.sin(angleY);
        const z2 = atom.x * Math.sin(angleY) + z1 * Math.cos(angleY);
        
        atom.x = x2;
        atom.y = y1;
        atom.z = z2;
      });

      // Clear
      ctx.fillStyle = "#070504";
      ctx.fillRect(0, 0, width, height);

      // Draw grid backdrop
      ctx.strokeStyle = "rgba(214, 60, 55, 0.03)";
      ctx.lineWidth = 1;
      for (let i = 0; i < width; i += 40) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, height);
        ctx.stroke();
      }
      for (let j = 0; j < height; j += 40) {
        ctx.beginPath();
        ctx.moveTo(0, j);
        ctx.lineTo(width, j);
        ctx.stroke();
      }

      const centerX = width / 2;
      const centerY = height / 2;

      // Draw bonds
      ctx.strokeStyle = "rgba(199, 62, 59, 0.4)";
      bonds.forEach(bond => {
        const a1 = atoms[bond.source];
        const a2 = atoms[bond.target];
        if (!a1 || !a2) return;
        
        // Perspective factor
        const fov = 150;
        const scale1 = fov / (fov + a1.z);
        const scale2 = fov / (fov + a2.z);
        
        ctx.beginPath();
        ctx.moveTo(centerX + a1.x * scale1, centerY + a1.y * scale1);
        ctx.lineTo(centerX + a2.x * scale2, centerY + a2.y * scale2);
        ctx.lineWidth = 2 * ((scale1 + scale2) / 2);
        ctx.stroke();
      });

      // Draw atoms
      atoms.forEach(atom => {
        const fov = 150;
        const scale = fov / (fov + atom.z);
        const x2d = centerX + atom.x * scale;
        const y2d = centerY + atom.y * scale;
        const radius = 8 * scale;

        // Atom colors
        let color = "#c7b5aa"; // default Carbon-like gray
        if (atom.element === "O") color = "#e65840"; // oxygen red
        if (atom.element === "N") color = "#3b82f6"; // nitrogen blue
        if (atom.element === "H") color = "#f3f4f6"; // hydrogen white
        if (atom.element === "F") color = "#10b981"; // fluorine green
        if (atom.element === "S") color = "#fbbf24"; // sulfur yellow

        // Draw shadow/glow
        ctx.beginPath();
        ctx.arc(x2d, y2d, radius + 4, 0, Math.PI * 2);
        ctx.fillStyle = color + "22";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x2d, y2d, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Label
        ctx.fillStyle = atom.z < 0 ? "#ffffff" : "#070504";
        ctx.font = `bold ${Math.max(6, 9 * scale)}px Inter`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(atom.element, x2d, y2d);
      });

      animationRef.current = requestAnimationFrame(runSimulation);
    };

    runSimulation();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [activeTab, sandboxSmiles]);

  return (
    <div className="flex-1 w-full max-w-7xl mx-auto px-4 py-8 md:py-12 flex flex-col font-sans select-none">
      
      {/* Header */}
      <header className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-panel-border pb-6 mb-8 gap-4 animate-fade-in-up">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <Activity className="w-8 h-8 text-brand animate-pulse" />
            <h1 className="text-3xl font-extrabold text-text-primary tracking-tight glow-text font-display">
              AEGIS DFI
            </h1>
            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-brand/10 border border-brand/20 text-brand">
              AI Prediction
            </span>
          </div>
          <p className="text-sm text-text-secondary">
            AI-powered Drug-Food Interaction modeling and biochemical explainability dashboard
          </p>
        </div>
        
        {/* Navigation Tabs */}
        <div className="flex p-1 bg-bg-secondary border border-panel-border rounded-full shadow-inner gap-1">
          <button 
            onClick={() => setActiveTab("checker")}
            className={`px-5 py-2 text-xs font-semibold rounded-full transition-all cursor-pointer ${activeTab === "checker" ? "bg-brand text-white shadow-md" : "text-text-secondary hover:text-text-primary"}`}
          >
            Interaction Checker
          </button>
          <button 
            onClick={() => setActiveTab("sandbox")}
            className={`px-5 py-2 text-xs font-semibold rounded-full transition-all cursor-pointer ${activeTab === "sandbox" ? "bg-brand text-white shadow-md" : "text-text-secondary hover:text-text-primary"}`}
          >
            Molecular Sandbox
          </button>
          <button 
            onClick={() => setActiveTab("history")}
            className={`px-5 py-2 text-xs font-semibold rounded-full transition-all cursor-pointer ${activeTab === "history" ? "bg-brand text-white shadow-md" : "text-text-secondary hover:text-text-primary"}`}
          >
            Prediction Log
          </button>
          <button 
            onClick={() => setActiveTab("admin")}
            className={`px-5 py-2 text-xs font-semibold rounded-full transition-all cursor-pointer ${activeTab === "admin" ? "bg-brand text-white shadow-md" : "text-text-secondary hover:text-text-primary"}`}
          >
            Admin Panel
          </button>
        </div>
      </header>

      {/* Main content body */}
      <main className="flex-1 w-full grid grid-cols-1 gap-8 animate-fade-in-up">
        
        {/* Tab 1: Interaction Checker */}
        {activeTab === "checker" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* Input form */}
            <div className="lg:col-span-5 flex flex-col gap-6">
              <div className="glass-panel p-6 flex flex-col gap-5">
                <div className="flex items-center gap-2 border-b border-panel-border pb-3 mb-2">
                  <Brain className="w-5 h-5 text-brand" />
                  <h2 className="text-lg font-bold text-text-primary font-display">Target Chemical Ingestor</h2>
                </div>

                {/* Drug Search */}
                <div className="relative">
                  <label className="text-xs font-bold text-text-secondary uppercase tracking-wider block mb-1">
                    Select Medication / Drug
                  </label>
                  <div className="flex items-center bg-bg-secondary border border-panel-border rounded-lg px-3 py-2 focus-within:border-brand transition-all">
                    <Search className="w-4 h-4 text-text-muted mr-2" />
                    <input 
                      type="text" 
                      value={drugInput}
                      onChange={(e) => {
                        setDrugInput(e.target.value);
                        setDrugSmiles("");
                      }}
                      placeholder="e.g. Warfarin, Atorvastatin..." 
                      className="bg-transparent w-full text-sm text-text-primary placeholder:text-text-muted outline-none"
                    />
                  </div>
                  {showDrugDropdown && drugSuggestions.length > 0 && (
                    <div className="absolute z-10 w-full bg-bg-secondary border border-panel-border rounded-lg mt-1 max-h-48 overflow-y-auto shadow-lg">
                      {drugSuggestions.map(d => (
                        <div 
                          key={d.id}
                          onClick={() => {
                            setDrugInput(d.name);
                            setDrugSmiles(d.smiles);
                            setShowDrugDropdown(false);
                          }}
                          className="px-4 py-2 hover:bg-bg-tertiary text-sm text-text-secondary hover:text-text-primary cursor-pointer border-b border-panel-border last:border-b-0"
                        >
                          <div className="font-semibold">{d.name}</div>
                          <div className="text-[10px] text-text-muted truncate font-mono">{d.smiles}</div>
                        </div>
                      ))}
                    </div>
                  )}
                  {drugSmiles && (
                    <div className="mt-1 text-[10px] font-mono text-brand truncate">
                      SMILES: {drugSmiles}
                    </div>
                  )}
                </div>

                {/* Food Search */}
                <div className="relative">
                  <label className="text-xs font-bold text-text-secondary uppercase tracking-wider block mb-1">
                    Select Food / Beverage
                  </label>
                  <div className="flex items-center bg-bg-secondary border border-panel-border rounded-lg px-3 py-2 focus-within:border-brand transition-all">
                    <Search className="w-4 h-4 text-text-muted mr-2" />
                    <input 
                      type="text" 
                      value={foodInput}
                      onChange={(e) => {
                        setFoodInput(e.target.value);
                        setFoodSmiles("");
                      }}
                      placeholder="e.g. Spinach, Grapefruit, Milk..." 
                      className="bg-transparent w-full text-sm text-text-primary placeholder:text-text-muted outline-none"
                    />
                  </div>
                  {showFoodDropdown && foodSuggestions.length > 0 && (
                    <div className="absolute z-10 w-full bg-bg-secondary border border-panel-border rounded-lg mt-1 max-h-48 overflow-y-auto shadow-lg">
                      {foodSuggestions.map(f => (
                        <div 
                          key={f.id}
                          onClick={() => {
                            setFoodInput(f.name);
                            setFoodSmiles(f.compounds[0]?.smiles || "");
                            setShowFoodDropdown(false);
                          }}
                          className="px-4 py-2 hover:bg-bg-tertiary text-sm text-text-secondary hover:text-text-primary cursor-pointer border-b border-panel-border last:border-b-0"
                        >
                          <div className="font-semibold">{f.name}</div>
                          <div className="text-[10px] text-text-muted truncate font-mono">{f.compounds[0]?.smiles || "No smiles"}</div>
                        </div>
                      ))}
                    </div>
                  )}
                  {foodSmiles && (
                    <div className="mt-1 text-[10px] font-mono text-brand truncate">
                      Bioactive SMILES: {foodSmiles}
                    </div>
                  )}
                </div>

                {/* Submit button */}
                <button 
                  onClick={runPrediction}
                  disabled={isAnalyzing || (!drugInput && !drugSmiles) || (!foodInput && !foodSmiles)}
                  className="w-full mt-4 btn-brand py-3 text-sm flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Analyzing Chemical Profiles...
                    </>
                  ) : (
                    <>
                      <Activity className="w-4 h-4" />
                      Check Drug-Food Interaction
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Results section */}
            <div className="lg:col-span-7 flex flex-col gap-6">
              
              {/* Prediction details card */}
              {predictionResult ? (
                <div className="glass-panel p-6 flex flex-col gap-6">
                  <div className="flex flex-col md:flex-row md:justify-between border-b border-panel-border pb-4 gap-4">
                    <div>
                      <h3 className="text-xl font-extrabold text-text-primary font-display mb-1">
                        Predictive Analysis Output
                      </h3>
                      <p className="text-xs text-text-muted">
                        Computed utilizing: {predictionResult.prediction.mode}
                      </p>
                    </div>
                    
                    {/* Severity Badge */}
                    <div className="flex items-center">
                      <span className={`px-4 py-2 rounded-full text-xs font-bold ${
                        predictionResult.prediction.severity === "severe" 
                          ? "bg-red-500/10 border border-red-500/30 text-red-500" 
                          : predictionResult.prediction.severity === "moderate"
                          ? "bg-amber-500/10 border border-amber-500/30 text-amber-500"
                          : "bg-green-500/10 border border-green-500/30 text-green-500"
                      }`}>
                        {predictionResult.prediction.badge}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
                    {/* Left: Confidence meter */}
                    <div className="md:col-span-4 flex flex-col items-center justify-center border-r border-panel-border pr-0 md:pr-6 gap-2">
                      <div className="relative w-28 h-28 flex items-center justify-center">
                        <svg className="w-full h-full transform -rotate-90">
                          <circle cx="56" cy="56" r="48" stroke="rgba(255,255,255,0.05)" strokeWidth="8" fill="transparent" />
                          <circle 
                            cx="56" cy="56" r="48" 
                            stroke={predictionResult.prediction.severity === 'severe' ? '#e65840' : predictionResult.prediction.severity === 'moderate' ? '#f0b054' : '#6cb88a'} 
                            strokeWidth="8" 
                            fill="transparent" 
                            strokeDasharray={2 * Math.PI * 48}
                            strokeDashoffset={2 * Math.PI * 48 * (1 - predictionResult.prediction.confidence)}
                          />
                        </svg>
                        <div className="absolute flex flex-col items-center">
                          <span className="text-2xl font-black font-display text-text-primary">
                            {(predictionResult.prediction.confidence * 100).toFixed(0)}%
                          </span>
                          <span className="text-[10px] text-text-muted uppercase tracking-wider font-semibold">
                            Confidence
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Right: Recommendation details */}
                    <div className="md:col-span-8 flex flex-col justify-between gap-4">
                      <div>
                        <h4 className="text-xs uppercase font-extrabold tracking-wider text-text-secondary mb-1 flex items-center gap-1.5">
                          <Info className="w-3.5 h-3.5 text-brand" /> Molecular Conflict Mechanism
                        </h4>
                        <p className="text-sm text-text-secondary leading-relaxed">
                          {predictionResult.clinical.mechanism}
                        </p>
                      </div>
                      
                      <div className="bg-bg-secondary/40 border border-panel-border p-4 rounded-lg">
                        <h4 className="text-xs uppercase font-extrabold tracking-wider text-text-secondary mb-1 flex items-center gap-1.5">
                          <UserCheck className="w-3.5 h-3.5 text-brand" /> Clinical Advice Recommendation
                        </h4>
                        <p className="text-xs text-text-primary italic leading-relaxed">
                          "{predictionResult.clinical.recommendation}"
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* SHAP EXPLAINABILITY PANEL */}
                  <div className="border-t border-panel-border pt-6 mt-2 flex flex-col gap-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs uppercase font-extrabold tracking-wider text-text-secondary flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-brand animate-pulse" /> Explainable AI (SHAP Descriptor Analysis)
                      </h4>
                      {isExplaining && <RefreshCw className="w-3 h-3 text-text-muted animate-spin" />}
                    </div>

                    {shapResult && shapResult.shap_available ? (
                      <div className="flex flex-col gap-4">
                        {/* Summary description */}
                        <div className="text-xs text-text-secondary border-l-2 border-brand/50 pl-3 leading-relaxed" 
                             dangerouslySetInnerHTML={{__html: shapResult.explanation.replace(/\n/g, '<br/>')}}>
                        </div>

                        {/* Top Features Chart */}
                        <div className="flex flex-col gap-2 mt-2">
                          <span className="text-[10px] text-text-muted uppercase tracking-wider block font-bold mb-1">
                            Top Contributing Descriptors (Impact on Class: {shapResult.predicted_label})
                          </span>
                          
                          <div className="flex flex-col gap-2">
                            {shapResult.features.slice(0, 5).map((f: any, idx: number) => {
                              const maxVal = Math.max(...shapResult.features.map((x: any) => Math.abs(x.shap_value)));
                              const pct = Math.min(100, (Math.abs(f.shap_value) / maxVal) * 100);
                              const isPositiveEffect = f.shap_value > 0;
                              
                              return (
                                <div key={idx} className="flex flex-col text-xs bg-bg-secondary/20 p-2 rounded border border-panel-border/50">
                                  <div className="flex justify-between font-medium text-text-secondary mb-1">
                                    <span className="truncate">{f.name}</span>
                                    <span className="font-mono text-[10px] text-text-muted">val: {f.value.toFixed(2)}</span>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <div className="w-16 text-[10px] text-text-muted truncate">
                                      {isPositiveEffect ? "Increases risk" : "Stabilizes"}
                                    </div>
                                    <div className="flex-1 bg-bg-secondary h-2.5 rounded-full overflow-hidden flex">
                                      <div 
                                        style={{ 
                                          width: `${pct}%`,
                                          backgroundColor: isPositiveEffect ? "#e65840" : "#6cb88a"
                                        }} 
                                        className="h-full rounded-full"
                                      />
                                    </div>
                                    <div className="w-10 text-right font-mono text-[10px] text-text-muted">
                                      {f.shap_value.toFixed(3)}
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-xs text-text-muted italic">
                        Loading SHAP explainability descriptors. Please wait...
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                <div className="glass-panel p-12 flex flex-col items-center justify-center text-center text-text-muted">
                  <Activity className="w-12 h-12 text-panel-border mb-4 animate-pulse" />
                  <p className="text-sm font-semibold text-text-secondary">
                    Ingestion Engine Idle
                  </p>
                  <p className="text-xs mt-1">
                    Select a drug and a food target compound in the Ingestor panel to trigger predictive analysis.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Molecular Sandbox */}
        {activeTab === "sandbox" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Control panel */}
            <div className="lg:col-span-5 flex flex-col gap-6">
              <div className="glass-panel p-6 flex flex-col gap-5">
                <div className="flex items-center gap-2 border-b border-panel-border pb-3 mb-2">
                  <Database className="w-5 h-5 text-brand" />
                  <h2 className="text-lg font-bold text-text-primary font-display">Bioactive SMILES Compiler</h2>
                </div>

                <div>
                  <label className="text-xs font-bold text-text-secondary uppercase tracking-wider block mb-1">
                    Molecular SMILES String
                  </label>
                  <textarea 
                    value={sandboxSmiles}
                    onChange={(e) => {
                      setSandboxSmiles(e.target.value);
                      setSandboxMoleculeName("Custom compound");
                    }}
                    rows={4}
                    className="w-full bg-bg-secondary border border-panel-border rounded-lg p-3 text-sm text-text-primary placeholder:text-text-muted font-mono outline-none focus:border-brand"
                    placeholder="Enter SMILES string (e.g. CCO)"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-[10px] font-bold text-text-muted uppercase tracking-wider block mb-1">
                      Compound Name
                    </label>
                    <input 
                      type="text" 
                      value={sandboxMoleculeName} 
                      onChange={(e) => setSandboxMoleculeName(e.target.value)}
                      className="w-full bg-bg-secondary border border-panel-border rounded-lg px-3 py-2 text-sm text-text-primary outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] font-bold text-text-muted uppercase tracking-wider block mb-1">
                      Chemical Formula
                    </label>
                    <input 
                      type="text" 
                      value={sandboxFormula} 
                      onChange={(e) => setSandboxFormula(e.target.value)}
                      className="w-full bg-bg-secondary border border-panel-border rounded-lg px-3 py-2 text-sm text-text-primary outline-none"
                    />
                  </div>
                </div>

                {/* Preset selectors */}
                <div>
                  <span className="text-[10px] font-bold text-text-muted uppercase tracking-wider block mb-2">
                    Chemical Preset Libraries
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { name: "Warfarin", smiles: "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O", formula: "C19H16O4", mw: 308.33 },
                      { name: "Caffeine", smiles: "CN1C(=O)N(C)C2=C1N=CN2C", formula: "C8H10N4O2", mw: 194.19 },
                      { name: "Bergamottin", smiles: "CC(=CCCC(=CC1=CC2=C(C=C1)OC(=O)C=C2)OC3=CC4=C(C=C3)C=CO4)C", formula: "C21H22O4", mw: 338.4 },
                      { name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O", formula: "C9H8O4", mw: 180.16 },
                      { name: "Ethanol", smiles: "CCO", formula: "C2H6O", mw: 46.07 }
                    ].map((p, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          setSandboxSmiles(p.smiles);
                          setSandboxMoleculeName(p.name);
                          setSandboxFormula(p.formula);
                          setSandboxWeight(p.mw);
                        }}
                        className="px-3 py-1.5 text-xs font-semibold rounded bg-bg-tertiary border border-panel-border hover:border-brand text-text-secondary hover:text-text-primary cursor-pointer transition-all"
                      >
                        {p.name}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Simulated 3D display panel */}
            <div className="lg:col-span-7">
              <div className="glass-panel p-6 flex flex-col gap-4 min-h-[460px]">
                <div className="flex items-center justify-between border-b border-panel-border pb-3">
                  <div>
                    <h3 className="text-lg font-bold text-text-primary font-display mb-0.5">
                      3D Molecular Layout Simulator
                    </h3>
                    <p className="text-xs text-text-muted">
                      Conformer coordinates computed using force-directed Hooke spring layout
                    </p>
                  </div>
                  <span className="text-xs font-semibold text-brand font-mono">
                    {sandboxFormula} | {sandboxWeight} g/mol
                  </span>
                </div>
                
                {/* 3D Canvas element wrapper */}
                <div className="flex-1 w-full bg-[#070504] border border-panel-border rounded-lg relative overflow-hidden flex items-center justify-center">
                  <canvas ref={canvasRef} className="w-full h-full max-h-[380px] block cursor-move" />
                  <div className="absolute top-3 left-3 bg-bg-secondary/80 border border-panel-border px-3 py-1.5 rounded text-left">
                    <span className="text-[10px] text-text-muted uppercase block font-bold">Bioactive Active</span>
                    <span className="text-xs font-semibold text-text-primary">{sandboxMoleculeName}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Prediction History logs */}
        {activeTab === "history" && (
          <div className="glass-panel p-6">
            <div className="flex items-center gap-2 border-b border-panel-border pb-3 mb-6">
              <ClipboardList className="w-5 h-5 text-brand" />
              <h2 className="text-lg font-bold text-text-primary font-display">System Prediction Logs</h2>
            </div>
            
            {isLoadingHistory ? (
              <div className="py-12 flex justify-center text-text-muted gap-2">
                <RefreshCw className="w-4 h-4 animate-spin text-brand" /> Loading logs...
              </div>
            ) : historyList.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-panel-border text-text-muted uppercase font-bold text-[10px] tracking-wider">
                      <th className="py-3 px-4">Medication (Drug)</th>
                      <th className="py-3 px-4">Diet Target (Food)</th>
                      <th className="py-3 px-4">AI Prediction Result</th>
                      <th className="py-3 px-4">Confidence</th>
                      <th className="py-3 px-4">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {historyList.map((h, idx) => (
                      <tr key={idx} className="border-b border-panel-border/30 hover:bg-bg-secondary/40 text-text-secondary hover:text-text-primary transition-all">
                        <td className="py-4 px-4 font-semibold">{h.drug_name}</td>
                        <td className="py-4 px-4 font-semibold">{h.food_name}</td>
                        <td className="py-4 px-4">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                            h.predicted_class === 0 
                              ? "bg-red-500/10 border-red-500/20 text-red-500" 
                              : h.predicted_class === 2
                              ? "bg-amber-500/10 border-amber-500/20 text-amber-500"
                              : "bg-green-500/10 border-green-500/20 text-green-500"
                          }`}>
                            {h.predicted_class === 0 ? "Negative DFI" : h.predicted_class === 2 ? "Caution/Non-sig" : "Safe DFI"}
                          </span>
                        </td>
                        <td className="py-4 px-4 font-mono text-[11px]">{(h.confidence * 100).toFixed(1)}%</td>
                        <td className="py-4 px-4 text-text-muted">{new Date(h.created_at).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-center py-12 text-text-muted italic">
                No predictions recorded in history. Run checking analysis to log runs.
              </p>
            )}
          </div>
        )}

        {/* Tab 4: Admin console */}
        {activeTab === "admin" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-fade-in-up">
            
            {/* Metadata and triggers */}
            <div className="lg:col-span-6 flex flex-col gap-6">
              
              {/* Active Model Metadata card */}
              <div className="glass-panel p-6">
                <div className="flex items-center gap-2 border-b border-panel-border pb-3 mb-4">
                  <Cpu className="w-5 h-5 text-brand" />
                  <h3 className="text-base font-bold text-text-primary font-display">Active Model Specifications</h3>
                </div>
                
                {modelMetadata ? (
                  <div className="flex flex-col gap-4 text-xs">
                    <div className="grid grid-cols-2 gap-4 bg-bg-secondary/40 p-4 rounded-lg border border-panel-border/30">
                      <div>
                        <span className="text-[10px] text-text-muted uppercase block font-bold">Classifier</span>
                        <span className="text-sm font-semibold text-text-primary">XGBoost Multiclass</span>
                      </div>
                      <div>
                        <span className="text-[10px] text-text-muted uppercase block font-bold">Model Version</span>
                        <span className="text-sm font-semibold text-brand font-mono">
                          {modelMetadata.version || "Unknown"}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <span className="text-[10px] text-text-muted uppercase block font-bold">Validation Accuracy</span>
                        <span className="text-sm font-semibold text-text-primary font-mono">
                          {modelMetadata.accuracy ? `${(modelMetadata.accuracy * 100).toFixed(2)}%` : "N/A"}
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] text-text-muted uppercase block font-bold">F1 weighted Score</span>
                        <span className="text-sm font-semibold text-text-primary font-mono">
                          {modelMetadata.f1_score ? `${(modelMetadata.f1_score * 100).toFixed(2)}%` : "N/A"}
                        </span>
                      </div>
                    </div>
                    
                    <div className="border-t border-panel-border/30 pt-4 mt-2">
                      <span className="text-[10px] text-text-muted uppercase block font-bold mb-2">Selected Feature Space (Top 18 Descriptors)</span>
                      <div className="flex flex-wrap gap-1.5">
                        {modelMetadata.features && modelMetadata.features.length > 0 ? (
                          modelMetadata.features.map((f: string, i: number) => (
                            <span key={i} className="px-2 py-0.5 rounded text-[10px] bg-bg-tertiary border border-panel-border/40 text-text-secondary font-mono">
                              {f}
                            </span>
                          ))
                        ) : (
                          <span className="text-text-muted italic">Model not trained. No features selected yet.</span>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-text-muted italic">Loading model specifications...</p>
                )}
              </div>

              {/* Seeding & Ingestion Stats */}
              <div className="glass-panel p-6">
                <div className="flex items-center gap-2 border-b border-panel-border pb-3 mb-4">
                  <Database className="w-5 h-5 text-brand" />
                  <h3 className="text-base font-bold text-text-primary font-display">Database Statistics</h3>
                </div>

                {platformStats ? (
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="bg-bg-secondary/40 p-4 rounded-lg border border-panel-border/30">
                      <span className="text-2xl font-black text-brand font-display">{platformStats.drugs}</span>
                      <span className="text-[10px] text-text-muted uppercase block font-bold mt-1">Drugs</span>
                    </div>
                    <div className="bg-bg-secondary/40 p-4 rounded-lg border border-panel-border/30">
                      <span className="text-2xl font-black text-brand font-display">{platformStats.foods}</span>
                      <span className="text-[10px] text-text-muted uppercase block font-bold mt-1">Foods</span>
                    </div>
                    <div className="bg-bg-secondary/40 p-4 rounded-lg border border-panel-border/30">
                      <span className="text-2xl font-black text-brand font-display">{platformStats.interactions_labeled}</span>
                      <span className="text-[10px] text-text-muted uppercase block font-bold mt-1">DFI Labels</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-text-muted italic">Loading statistics...</p>
                )}
              </div>
            </div>

            {/* Model retraining log console */}
            <div className="lg:col-span-6 flex flex-col gap-6">
              <div className="glass-panel p-6 flex flex-col gap-4 h-full min-h-[400px]">
                <div className="flex justify-between items-center border-b border-panel-border pb-3">
                  <h3 className="text-base font-bold text-text-primary font-display">Model Pipeline Retraining Console</h3>
                  <button 
                    onClick={runRetrain}
                    disabled={isRetraining || retrainStatus === "running"}
                    className="btn-brand px-4 py-2 text-xs flex items-center gap-1.5 cursor-pointer disabled:opacity-50"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${isRetraining ? 'animate-spin' : ''}`} />
                    Retrain Classifier Model
                  </button>
                </div>

                <div className="flex-1 w-full bg-bg-secondary p-4 rounded-lg border border-panel-border font-mono text-[10px] text-text-secondary overflow-y-auto max-h-[300px]">
                  <pre className="whitespace-pre-wrap">{retrainLogs || "No logs available. Trigger retraining to view execution logs."}</pre>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 pt-6 border-t border-panel-border text-center text-text-muted text-xs animate-fade-in-up">
        <p>© 2026 Aegis DFI Platform. Suitable for medical research and educational use. Descriptors engineered with RDKit. Classifier models powered by XGBoost. Explainability calculated with SHAP.</p>
      </footer>
    </div>
  );
}
