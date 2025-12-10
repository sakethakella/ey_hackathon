import React, { useState, useEffect } from 'react';
import { 
  Wrench, 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Search, 
  Save, 
  RefreshCw,
  Truck,
  FileText,
  Cpu
} from 'lucide-react';

// --- Configuration ---
const API_BASE = '/api/v1';
const WORKSHOP_ID = 'ws_001'; // Default workshop ID

// --- Mock Data for Preview (Fallback) ---
const MOCK_JOBS = [
  { id: 'job_102938', vehicle_id: 'v_tesla_x99', status: 'pending', issue: 'Vibration detected in rear axle', created_at: '2023-10-25T09:00:00Z' },
  { id: 'job_102939', vehicle_id: 'v_rivian_t1', status: 'in_progress', issue: 'Battery thermal runaway warning', created_at: '2023-10-25T10:15:00Z' },
  { id: 'job_102940', vehicle_id: 'v_ford_f150', status: 'completed', issue: 'Routine diagnostic check', created_at: '2023-10-24T14:30:00Z' },
];

const MOCK_DETAILS = {
  telemetry: {
    rpm: 3400,
    vibration_x: 0.04,
    vibration_y: 1.25, // Anomaly
    vibration_z: 0.02,
    temp_c: 85,
    battery_level: 45,
    timestamp: '2023-10-25T10:30:45Z'
  },
  anomalies: [
    { id: 'anom_1', timestamp: '2023-10-25T10:28:00Z', score: 0.95, sensor: 'vibration_y', description: 'High frequency oscillation detected' },
    { id: 'anom_2', timestamp: '2023-10-25T10:15:00Z', score: 0.88, sensor: 'temp_c', description: 'Temperature spike above threshold' }
  ],
  job_extended: {
    notes: 'Initial inspection shows loose mounting bracket on the sensor array.',
    ai_report: 'Gemini Analysis: The vibration pattern in the Y-axis correlates with a Type-B coupling failure with 89% confidence. Recommended action: Inspect rear coupling bolts.'
  }
};

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [loadingJobs, setLoadingJobs] = useState(false);
  
  // Detail State
  const [jobDetails, setJobDetails] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [notes, setNotes] = useState('');
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);

  // --- API Interactions ---

  const fetchJobs = async () => {
    setLoadingJobs(true);
    try {
      const res = await fetch(`${API_BASE}/workshops/${WORKSHOP_ID}/jobs`);
      if (!res.ok) throw new Error('API unreachable');
      const data = await res.json();
      setJobs(data);
      setIsDemoMode(false);
    } catch (err) {
      console.warn('Backend unavailable, switching to Demo Mode');
      setJobs(MOCK_JOBS);
      setIsDemoMode(true);
    } finally {
      setLoadingJobs(false);
    }
  };

  const fetchJobData = async (job) => {
    setLoadingDetails(true);
    setSelectedJobId(job.id);
    setJobDetails(null); // Clear previous

    try {
      if (isDemoMode) {
        // Simulate network delay
        await new Promise(r => setTimeout(r, 600));
        setJobDetails({ ...job, ...MOCK_DETAILS.job_extended });
        setTelemetry(MOCK_DETAILS.telemetry);
        setAnomalies(MOCK_DETAILS.anomalies);
        setNotes(MOCK_DETAILS.job_extended.notes);
      } else {
        // 1. Get Job Details
        const jobRes = await fetch(`${API_BASE}/jobs/${job.id}`);
        const jobData = await jobRes.json();
        setJobDetails(jobData);
        setNotes(jobData.notes || '');

        // 2. Get Telemetry
        const telRes = await fetch(`${API_BASE}/vehicles/${job.vehicle_id}/telemetry/latest`);
        const telData = await telRes.json();
        setTelemetry(telData);

        // 3. Get Anomalies
        const anomRes = await fetch(`${API_BASE}/vehicles/${job.vehicle_id}/anomalies`);
        const anomData = await anomRes.json();
        setAnomalies(anomData);
      }
    } catch (err) {
      console.error("Failed to load details", err);
    } finally {
      setLoadingDetails(false);
    }
  };

  const updateJob = async (status) => {
    if (isDemoMode) {
      setJobDetails(prev => ({ ...prev, status }));
      setJobs(prev => prev.map(j => j.id === selectedJobId ? { ...j, status } : j));
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/jobs/${selectedJobId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      });
      if (res.ok) {
        setJobDetails(prev => ({ ...prev, status }));
        setJobs(prev => prev.map(j => j.id === selectedJobId ? { ...j, status } : j));
      }
    } catch (err) {
      console.error("Failed to update status", err);
    }
  };

  const saveNotes = async () => {
    if (isDemoMode) {
      alert("Notes saved (Demo Mode)");
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/jobs/${selectedJobId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes })
      });
      if (res.ok) {
        alert("Notes saved successfully");
      }
    } catch (err) {
      console.error("Failed to save notes", err);
    }
  };

  // --- Effects ---

  useEffect(() => {
    fetchJobs();
    // Auto-refresh jobs every 30s
    const interval = setInterval(fetchJobs, 30000);
    return () => clearInterval(interval);
  }, []);

  // --- Render Helpers ---

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800 border-green-200';
      case 'in_progress': return 'bg-blue-100 text-blue-800 border-blue-200';
      default: return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-900">
      
      {/* --- Sidebar: Job List --- */}
      <div className="w-1/3 min-w-[350px] max-w-md bg-white border-r border-slate-200 flex flex-col shadow-sm z-10">
        
        {/* Sidebar Header */}
        <div className="p-5 border-b border-slate-200 bg-slate-900 text-white">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2 bg-indigo-500 rounded-lg">
              <Wrench className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">MotoVerse</h1>
              <p className="text-xs text-slate-400 font-mono tracking-wide">WORKSHOP CONSOLE</p>
            </div>
          </div>
          
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-3 text-slate-400" />
            <input 
              type="text" 
              placeholder="Search Vehicle ID or Job..." 
              className="w-full bg-slate-800 border border-slate-700 rounded-md pl-9 pr-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 placeholder-slate-500"
            />
          </div>
        </div>

        {/* Job List */}
        <div className="flex-1 overflow-y-auto">
          {loadingJobs && jobs.length === 0 ? (
            <div className="p-8 text-center text-slate-500">Loading jobs...</div>
          ) : (
            <ul className="divide-y divide-slate-100">
              {jobs.map(job => (
                <li 
                  key={job.id} 
                  onClick={() => fetchJobData(job)}
                  className={`p-4 cursor-pointer transition-colors hover:bg-slate-50 ${selectedJobId === job.id ? 'bg-indigo-50 border-l-4 border-indigo-500' : 'border-l-4 border-transparent'}`}
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-mono text-xs font-semibold text-slate-500">#{job.id.slice(-6)}</span>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${getStatusColor(job.status)}`}>
                      {job.status.replace('_', ' ')}
                    </span>
                  </div>
                  <h3 className="font-semibold text-slate-800 mb-1 flex items-center gap-2">
                    <Truck className="w-4 h-4 text-slate-400" />
                    {job.vehicle_id}
                  </h3>
                  <p className="text-sm text-slate-600 line-clamp-2">{job.issue}</p>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 text-xs text-slate-500 flex justify-between items-center">
          <span>{jobs.length} Active Jobs</span>
          <button onClick={fetchJobs} className="flex items-center gap-1 hover:text-indigo-600">
            <RefreshCw className="w-3 h-3" /> Refresh
          </button>
        </div>
      </div>

      {/* --- Main Content: Job Details --- */}
      <div className="flex-1 overflow-y-auto bg-slate-50/50">
        {!selectedJobId ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-400">
            <Wrench className="w-16 h-16 mb-4 opacity-20" />
            <p className="text-lg font-medium">Select a job to view details</p>
          </div>
        ) : loadingDetails || !jobDetails ? (
          <div className="h-full flex items-center justify-center">
             <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          </div>
        ) : (
          <div className="max-w-5xl mx-auto p-6 space-y-6">
            
            {/* Header Card */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 flex justify-between items-start">
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <h2 className="text-2xl font-bold text-slate-900">{jobDetails.vehicle_id}</h2>
                  <span className={`px-3 py-1 rounded-full text-sm font-semibold border ${getStatusColor(jobDetails.status)} uppercase tracking-wide`}>
                    {jobDetails.status.replace('_', ' ')}
                  </span>
                </div>
                <p className="text-slate-600 max-w-2xl text-lg">{jobDetails.issue}</p>
                <div className="mt-4 flex items-center gap-4 text-sm text-slate-500 font-mono">
                  <span className="flex items-center gap-1"><FileText className="w-4 h-4" /> Job ID: {jobDetails.id}</span>
                  <span className="flex items-center gap-1"><Clock className="w-4 h-4" /> Created: {new Date(jobDetails.created_at || Date.now()).toLocaleDateString()}</span>
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <button 
                  onClick={() => updateJob('in_progress')}
                  disabled={jobDetails.status === 'in_progress'}
                  className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium shadow-sm transition-all"
                >
                  <Activity className="w-4 h-4" /> Mark In Progress
                </button>
                <button 
                  onClick={() => updateJob('completed')}
                  disabled={jobDetails.status === 'completed'}
                  className="flex items-center justify-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg font-medium shadow-sm transition-all"
                >
                  <CheckCircle className="w-4 h-4" /> Mark Completed
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* Left Column: Diagnostics & AI */}
              <div className="lg:col-span-2 space-y-6">
                
                {/* AI Analysis Card */}
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                  <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-purple-600" />
                    AI Agent Analysis
                  </h3>
                  <div className="bg-purple-50 border border-purple-100 rounded-lg p-4">
                    <p className="text-slate-800 leading-relaxed whitespace-pre-wrap">
                      {jobDetails.ai_report || "Waiting for agent analysis..."}
                    </p>
                  </div>
                </div>

                {/* Telemetry Panel */}
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                      <Activity className="w-4 h-4 text-indigo-600" />
                      Live Telemetry Snapshot
                    </h3>
                    <span className="text-xs font-mono text-slate-400">
                      Last Update: {telemetry?.timestamp ? new Date(telemetry.timestamp).toLocaleTimeString() : 'N/A'}
                    </span>
                  </div>
                  
                  {/* Visual Quick Stats */}
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="bg-slate-50 p-3 rounded border border-slate-100 text-center">
                      <div className="text-xs text-slate-500 uppercase">RPM</div>
                      <div className="text-xl font-bold text-slate-900">{telemetry?.rpm || '--'}</div>
                    </div>
                    <div className="bg-slate-50 p-3 rounded border border-slate-100 text-center">
                      <div className="text-xs text-slate-500 uppercase">Temp</div>
                      <div className={`text-xl font-bold ${telemetry?.temp_c > 90 ? 'text-red-600' : 'text-slate-900'}`}>
                        {telemetry?.temp_c ? `${telemetry.temp_c}°C` : '--'}
                      </div>
                    </div>
                    <div className="bg-slate-50 p-3 rounded border border-slate-100 text-center">
                      <div className="text-xs text-slate-500 uppercase">Battery</div>
                      <div className="text-xl font-bold text-slate-900">{telemetry?.battery_level ? `${telemetry.battery_level}%` : '--'}</div>
                    </div>
                  </div>

                  {/* Raw JSON */}
                  <div className="relative">
                     <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg text-xs font-mono overflow-x-auto border border-slate-800">
{JSON.stringify(telemetry || {}, null, 2)}
                    </pre>
                  </div>
                </div>

                {/* Anomaly History */}
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
                  <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-500" />
                    Anomaly History
                  </h3>
                  {anomalies.length > 0 ? (
                    <div className="overflow-hidden rounded-lg border border-slate-200">
                      <table className="w-full text-sm text-left">
                        <thead className="bg-slate-50 text-slate-500 font-semibold">
                          <tr>
                            <th className="px-4 py-3">Time</th>
                            <th className="px-4 py-3">Sensor</th>
                            <th className="px-4 py-3">Score</th>
                            <th className="px-4 py-3">Note</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {anomalies.map((anom, idx) => (
                            <tr key={idx} className="hover:bg-slate-50">
                              <td className="px-4 py-3 font-mono text-xs">{new Date(anom.timestamp).toLocaleTimeString()}</td>
                              <td className="px-4 py-3">{anom.sensor}</td>
                              <td className="px-4 py-3">
                                <span className={`px-2 py-0.5 rounded text-xs font-bold ${anom.score > 0.9 ? 'bg-red-100 text-red-700' : 'bg-yellow-100 text-yellow-700'}`}>
                                  {anom.score.toFixed(2)}
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-600">{anom.description || '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-sm text-slate-500 italic">No historical anomalies detected.</div>
                  )}
                </div>

              </div>

              {/* Right Column: Mechanic Notes */}
              <div className="lg:col-span-1">
                <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 sticky top-6">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">Mechanic Notes</h3>
                    <button 
                      onClick={saveNotes}
                      className="text-xs flex items-center gap-1 bg-indigo-50 text-indigo-700 px-2 py-1 rounded hover:bg-indigo-100 transition-colors"
                    >
                      <Save className="w-3 h-3" /> Save
                    </button>
                  </div>
                  <textarea
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    className="w-full h-96 p-4 bg-yellow-50/50 border border-yellow-200 rounded-lg text-sm text-slate-800 leading-relaxed focus:outline-none focus:ring-2 focus:ring-yellow-400 resize-none font-mono"
                    placeholder="Enter diagnosis notes, replacement parts, or observations..."
                  ></textarea>
                  <p className="mt-2 text-xs text-slate-400">
                    Auto-saved locally. Press Save to commit to server.
                  </p>
                </div>
              </div>

            </div>
          </div>
        )}
      </div>

      {isDemoMode && (
        <div className="fixed bottom-4 right-4 bg-amber-100 border border-amber-300 text-amber-800 px-3 py-1.5 rounded-full text-xs font-bold shadow-lg z-50">
          ⚠️ Demo Mode: Backend Unavailable
        </div>
      )}

    </div>
  );
}
