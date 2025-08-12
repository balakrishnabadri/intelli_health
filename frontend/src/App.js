import React, { useState, useEffect } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, Link, useNavigate } from "react-router-dom";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Navigation Component
const Navigation = () => {
  return (
    <nav className="bg-blue-600 text-white p-4 shadow-lg">
      <div className="container mx-auto flex justify-between items-center">
        <Link to="/" className="text-2xl font-bold">
          🏥 IntelliHealth
        </Link>
        <div className="space-x-6">
          <Link to="/" className="hover:text-blue-200 transition-colors">Home</Link>
          <Link to="/predict" className="hover:text-blue-200 transition-colors">Predict</Link>
          <Link to="/history" className="hover:text-blue-200 transition-colors">History</Link>
          <Link to="/about" className="hover:text-blue-200 transition-colors">About</Link>
        </div>
      </div>
    </nav>
  );
};

// Home Component
const Home = () => {
  const [stats, setStats] = useState(null);
  
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get(`${API}/predictions/stats`);
        setStats(response.data);
      } catch (error) {
        console.error("Error fetching stats:", error);
      }
    };
    
    fetchStats();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-white">
      <div className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-800 mb-6">
            IntelliHealth Multi-Disease Prediction System
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Advanced AI-powered health prediction system that analyzes your medical data 
            to assess risk levels for multiple diseases including Diabetes, Heart Disease, and Parkinson's.
          </p>
          <Link
            to="/predict"
            className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors inline-block"
          >
            Start Health Assessment
          </Link>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-white p-8 rounded-lg shadow-lg text-center">
            <div className="text-4xl mb-4">🩺</div>
            <h3 className="text-xl font-semibold mb-3">Diabetes Prediction</h3>
            <p className="text-gray-600">
              Assess diabetes risk using glucose levels, BMI, age, and other key factors.
            </p>
          </div>
          <div className="bg-white p-8 rounded-lg shadow-lg text-center">
            <div className="text-4xl mb-4">❤️</div>
            <h3 className="text-xl font-semibold mb-3">Heart Disease</h3>
            <p className="text-gray-600">
              Evaluate cardiovascular risk based on blood pressure, cholesterol, and lifestyle factors.
            </p>
          </div>
          <div className="bg-white p-8 rounded-lg shadow-lg text-center">
            <div className="text-4xl mb-4">🧠</div>
            <h3 className="text-xl font-semibold mb-3">Parkinson's Disease</h3>
            <p className="text-gray-600">
              Analyze vocal patterns and neurological indicators for early detection.
            </p>
          </div>
        </div>

        {/* Stats */}
        {stats && (
          <div className="bg-white p-8 rounded-lg shadow-lg">
            <h3 className="text-2xl font-semibold mb-6 text-center">System Statistics</h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">{stats.total_predictions}</div>
                <div className="text-gray-600">Total Predictions</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600">
                  {stats.by_disease ? Object.keys(stats.by_disease).length : 0}
                </div>
                <div className="text-gray-600">Disease Types</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600">AI-Powered</div>
                <div className="text-gray-600">ML Models</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Disease Selection Component
const DiseaseSelection = () => {
  const navigate = useNavigate();
  
  const diseases = [
    {
      id: 'diabetes',
      name: 'Diabetes',
      icon: '🩺',
      description: 'Predict diabetes risk based on medical indicators',
      color: 'bg-red-500'
    },
    {
      id: 'heart',
      name: 'Heart Disease',
      icon: '❤️',
      description: 'Assess cardiovascular disease risk',
      color: 'bg-pink-500'
    },
    {
      id: 'parkinsons',
      name: "Parkinson's Disease",
      icon: '🧠',
      description: 'Analyze neurological patterns for early detection',
      color: 'bg-purple-500'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h2 className="text-4xl font-bold text-gray-800 mb-4">
            Select Disease for Prediction
          </h2>
          <p className="text-xl text-gray-600">
            Choose the disease you'd like to assess your risk for
          </p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
          {diseases.map((disease) => (
            <div
              key={disease.id}
              onClick={() => navigate(`/predict/${disease.id}`)}
              className="bg-white p-8 rounded-lg shadow-lg cursor-pointer hover:shadow-xl transition-shadow"
            >
              <div className="text-center">
                <div className="text-6xl mb-4">{disease.icon}</div>
                <h3 className="text-2xl font-semibold mb-3">{disease.name}</h3>
                <p className="text-gray-600 mb-6">{disease.description}</p>
                <button
                  className={`${disease.color} text-white px-6 py-3 rounded-lg font-semibold hover:opacity-90 transition-opacity`}
                >
                  Start Assessment
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Diabetes Form Component
const DiabetesForm = () => {
  const [formData, setFormData] = useState({
    pregnancies: 0,
    glucose: 120,
    blood_pressure: 80,
    skin_thickness: 20,
    insulin: 80,
    bmi: 25,
    diabetes_pedigree: 0.5,
    age: 30
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await axios.post(`${API}/predict/diabetes`, formData);
      setResult(response.data);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Error making prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  if (result) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="container mx-auto px-4 max-w-2xl">
          <div className="bg-white p-8 rounded-lg shadow-lg">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-4">
                Diabetes Risk Assessment Results
              </h2>
              <div className={`text-6xl mb-4 ${
                result.risk_level === 'High Risk' ? 'text-red-500' :
                result.risk_level === 'Moderate Risk' ? 'text-yellow-500' : 'text-green-500'
              }`}>
                {result.prediction === 1 ? '⚠️' : '✅'}
              </div>
              <div className={`text-2xl font-bold mb-4 ${
                result.risk_level === 'High Risk' ? 'text-red-500' :
                result.risk_level === 'Moderate Risk' ? 'text-yellow-500' : 'text-green-500'
              }`}>
                {result.risk_level}
              </div>
              <p className="text-xl text-gray-600 mb-6">
                Probability: {(result.probability * 100).toFixed(1)}%
              </p>
            </div>
            
            <div className="mb-8">
              <h3 className="text-xl font-semibold mb-4">Interpretation:</h3>
              <p className="text-gray-700 mb-4">
                {result.prediction === 1 
                  ? "Based on your inputs, the model indicates an elevated risk for diabetes. Please consult with a healthcare professional for proper medical evaluation."
                  : "Based on your inputs, the model indicates a lower risk for diabetes. However, this is not a medical diagnosis and regular health checkups are recommended."
                }
              </p>
            </div>
            
            <div className="flex gap-4">
              <button
                onClick={() => setResult(null)}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                New Assessment
              </button>
              <Link
                to="/predict"
                className="bg-gray-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-gray-700 transition-colors"
              >
                Other Diseases
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4 max-w-2xl">
        <div className="bg-white p-8 rounded-lg shadow-lg">
          <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">
            🩺 Diabetes Risk Assessment
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Pregnancies
                </label>
                <input
                  type="number"
                  name="pregnancies"
                  value={formData.pregnancies}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0"
                  max="20"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Glucose Level (mg/dL)
                </label>
                <input
                  type="number"
                  name="glucose"
                  value={formData.glucose}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0"
                  max="300"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Blood Pressure (mmHg)
                </label>
                <input
                  type="number"
                  name="blood_pressure"
                  value={formData.blood_pressure}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0"
                  max="200"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Skin Thickness (mm)
                </label>
                <input
                  type="number"
                  name="skin_thickness"
                  value={formData.skin_thickness}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0"
                  max="100"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Insulin (μU/mL)
                </label>
                <input
                  type="number"
                  name="insulin"
                  value={formData.insulin}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0"
                  max="900"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  BMI
                </label>
                <input
                  type="number"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleChange}
                  step="0.1"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0"
                  max="70"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Diabetes Pedigree Function
                </label>
                <input
                  type="number"
                  name="diabetes_pedigree"
                  value={formData.diabetes_pedigree}
                  onChange={handleChange}
                  step="0.01"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0"
                  max="3"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Age
                </label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0"
                  max="120"
                />
              </div>
            </div>
            
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {loading ? "Analyzing..." : "Predict Diabetes Risk"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

// Heart Disease Form Component
const HeartDiseaseForm = () => {
  const [formData, setFormData] = useState({
    age: 50,
    sex: 1,
    chest_pain_type: 0,
    resting_bp: 120,
    cholesterol: 200,
    fasting_bs: 0,
    resting_ecg: 0,
    max_hr: 150,
    exercise_angina: 0,
    oldpeak: 1.0,
    st_slope: 1
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await axios.post(`${API}/predict/heart`, formData);
      setResult(response.data);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Error making prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  if (result) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="container mx-auto px-4 max-w-2xl">
          <div className="bg-white p-8 rounded-lg shadow-lg">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-4">
                Heart Disease Risk Assessment Results
              </h2>
              <div className={`text-6xl mb-4 ${
                result.risk_level === 'High Risk' ? 'text-red-500' :
                result.risk_level === 'Moderate Risk' ? 'text-yellow-500' : 'text-green-500'
              }`}>
                {result.prediction === 1 ? '⚠️' : '✅'}
              </div>
              <div className={`text-2xl font-bold mb-4 ${
                result.risk_level === 'High Risk' ? 'text-red-500' :
                result.risk_level === 'Moderate Risk' ? 'text-yellow-500' : 'text-green-500'
              }`}>
                {result.risk_level}
              </div>
              <p className="text-xl text-gray-600 mb-6">
                Probability: {(result.probability * 100).toFixed(1)}%
              </p>
            </div>
            
            <div className="flex gap-4">
              <button
                onClick={() => setResult(null)}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                New Assessment
              </button>
              <Link
                to="/predict"
                className="bg-gray-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-gray-700 transition-colors"
              >
                Other Diseases
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4 max-w-2xl">
        <div className="bg-white p-8 rounded-lg shadow-lg">
          <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">
            ❤️ Heart Disease Risk Assessment
          </h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="120"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Sex</label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={0}>Female</option>
                  <option value={1}>Male</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Chest Pain Type</label>
                <select
                  name="chest_pain_type"
                  value={formData.chest_pain_type}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={0}>Typical Angina</option>
                  <option value={1}>Atypical Angina</option>
                  <option value={2}>Non-Anginal Pain</option>
                  <option value={3}>Asymptomatic</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Resting BP (mmHg)</label>
                <input
                  type="number"
                  name="resting_bp"
                  value={formData.resting_bp}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="250"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Cholesterol (mg/dL)</label>
                <input
                  type="number"
                  name="cholesterol"
                  value={formData.cholesterol}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="600"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Fasting Blood Sugar</label>
                <select
                  name="fasting_bs"
                  value={formData.fasting_bs}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={0}>≤ 120 mg/dL</option>
                  <option value={1}>> 120 mg/dL</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Max Heart Rate</label>
                <input
                  type="number"
                  name="max_hr"
                  value={formData.max_hr}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="250"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Exercise Angina</label>
                <select
                  name="exercise_angina"
                  value={formData.exercise_angina}
                  onChange={handleChange}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>
            </div>
            
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {loading ? "Analyzing..." : "Predict Heart Disease Risk"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

// Simplified Parkinson's Form (with fewer fields for better UX)
const ParkinsonsForm = () => {
  const [formData, setFormData] = useState({
    mdvp_fo: 150,
    mdvp_fhi: 200,
    mdvp_flo: 100,
    mdvp_jitter_percent: 0.5,
    mdvp_jitter_abs: 0.00003,
    mdvp_rap: 0.003,
    mdvp_ppq: 0.003,
    jitter_ddp: 0.009,
    mdvp_shimmer: 0.03,
    mdvp_shimmer_db: 0.3,
    shimmer_apq3: 0.015,
    shimmer_apq5: 0.018,
    mdvp_apq: 0.024,
    shimmer_dda: 0.045,
    nhr: 0.02,
    hnr: 25,
    rpde: 0.5,
    dfa: 0.7,
    spread1: -6,
    spread2: 0.2,
    d2: 2,
    ppe: 0.2
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await axios.post(`${API}/predict/parkinsons`, formData);
      setResult(response.data);
    } catch (error) {
      console.error("Prediction error:", error);
      alert("Error making prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  if (result) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="container mx-auto px-4 max-w-2xl">
          <div className="bg-white p-8 rounded-lg shadow-lg">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-4">
                Parkinson's Disease Risk Assessment Results
              </h2>
              <div className={`text-6xl mb-4 ${
                result.risk_level === 'High Risk' ? 'text-red-500' :
                result.risk_level === 'Moderate Risk' ? 'text-yellow-500' : 'text-green-500'
              }`}>
                {result.prediction === 1 ? '⚠️' : '✅'}
              </div>
              <div className={`text-2xl font-bold mb-4 ${
                result.risk_level === 'High Risk' ? 'text-red-500' :
                result.risk_level === 'Moderate Risk' ? 'text-yellow-500' : 'text-green-500'
              }`}>
                {result.risk_level}
              </div>
              <p className="text-xl text-gray-600 mb-6">
                Probability: {(result.probability * 100).toFixed(1)}%
              </p>
            </div>
            
            <div className="flex gap-4">
              <button
                onClick={() => setResult(null)}
                className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                New Assessment
              </button>
              <Link
                to="/predict"
                className="bg-gray-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-gray-700 transition-colors"
              >
                Other Diseases
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4 max-w-2xl">
        <div className="bg-white p-8 rounded-lg shadow-lg">
          <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">
            🧠 Parkinson's Disease Assessment
          </h2>
          <p className="text-center text-gray-600 mb-8">
            This assessment analyzes vocal patterns. For demonstration, we're using simplified values.
          </p>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Average Vocal Frequency (Hz)
                </label>
                <input
                  type="number"
                  name="mdvp_fo"
                  value={formData.mdvp_fo}
                  onChange={handleChange}
                  step="0.1"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="300"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Maximum Vocal Frequency (Hz)
                </label>
                <input
                  type="number"
                  name="mdvp_fhi"
                  value={formData.mdvp_fhi}
                  onChange={handleChange}
                  step="0.1"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="600"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Jitter Percentage (%)
                </label>
                <input
                  type="number"
                  name="mdvp_jitter_percent"
                  value={formData.mdvp_jitter_percent}
                  onChange={handleChange}
                  step="0.01"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="10"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Shimmer (%)
                </label>
                <input
                  type="number"
                  name="mdvp_shimmer"
                  value={formData.mdvp_shimmer}
                  onChange={handleChange}
                  step="0.001"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="1"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Noise-to-Harmonics Ratio (NHR)
                </label>
                <input
                  type="number"
                  name="nhr"
                  value={formData.nhr}
                  onChange={handleChange}
                  step="0.001"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="1"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Harmonics-to-Noise Ratio (HNR)
                </label>
                <input
                  type="number"
                  name="hnr"
                  value={formData.hnr}
                  onChange={handleChange}
                  step="0.1"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  min="0" max="50"
                />
              </div>
            </div>
            
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 text-white py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {loading ? "Analyzing..." : "Predict Parkinson's Risk"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

// History Component
const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await axios.get(`${API}/predictions/history`);
        setHistory(response.data);
      } catch (error) {
        console.error("Error fetching history:", error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchHistory();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-xl">Loading history...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4">
        <h2 className="text-4xl font-bold text-gray-800 mb-8 text-center">
          Prediction History
        </h2>
        
        {history.length === 0 ? (
          <div className="text-center">
            <p className="text-xl text-gray-600 mb-8">No predictions yet.</p>
            <Link
              to="/predict"
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Make Your First Prediction
            </Link>
          </div>
        ) : (
          <div className="grid gap-6 max-w-4xl mx-auto">
            {history.map((prediction) => (
              <div key={prediction.id} className="bg-white p-6 rounded-lg shadow-lg">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-xl font-semibold capitalize">
                      {prediction.disease_type.replace('_', ' ')} Assessment
                    </h3>
                    <p className="text-gray-600">
                      {new Date(prediction.timestamp).toLocaleDateString()} at{' '}
                      {new Date(prediction.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                  <div className={`px-4 py-2 rounded-full text-sm font-semibold ${
                    prediction.risk_level === 'High Risk' ? 'bg-red-100 text-red-800' :
                    prediction.risk_level === 'Moderate Risk' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {prediction.risk_level}
                  </div>
                </div>
                <p className="text-gray-700">
                  Probability: {(prediction.probability * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// About Component
const About = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="container mx-auto px-4 max-w-4xl">
        <h2 className="text-4xl font-bold text-gray-800 mb-8 text-center">
          About IntelliHealth
        </h2>
        
        <div className="bg-white p-8 rounded-lg shadow-lg mb-8">
          <h3 className="text-2xl font-semibold mb-4">Our Mission</h3>
          <p className="text-gray-700 mb-6">
            IntelliHealth is an advanced AI-powered health prediction system designed to help individuals 
            assess their risk for multiple diseases using machine learning algorithms trained on medical data.
          </p>
          
          <h3 className="text-2xl font-semibold mb-4">How It Works</h3>
          <div className="grid md:grid-cols-3 gap-6 mb-6">
            <div className="text-center">
              <div className="text-3xl mb-2">📊</div>
              <h4 className="font-semibold mb-2">Data Input</h4>
              <p className="text-sm text-gray-600">
                Enter your medical parameters through our user-friendly forms
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">🤖</div>
              <h4 className="font-semibold mb-2">AI Analysis</h4>
              <p className="text-sm text-gray-600">
                Our trained ML models analyze your data and calculate risk probabilities
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">📋</div>
              <h4 className="font-semibold mb-2">Risk Assessment</h4>
              <p className="text-sm text-gray-600">
                Receive detailed risk levels and recommendations
              </p>
            </div>
          </div>
          
          <h3 className="text-2xl font-semibold mb-4">Supported Conditions</h3>
          <ul className="list-disc list-inside text-gray-700 space-y-2">
            <li><strong>Diabetes:</strong> Risk assessment based on glucose levels, BMI, and family history</li>
            <li><strong>Heart Disease:</strong> Cardiovascular risk analysis using blood pressure, cholesterol, and lifestyle factors</li>
            <li><strong>Parkinson's Disease:</strong> Early detection through vocal pattern analysis</li>
          </ul>
          
          <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <h4 className="font-semibold text-yellow-800 mb-2">Important Disclaimer</h4>
            <p className="text-yellow-700 text-sm">
              This system is for educational and informational purposes only. It does not replace 
              professional medical advice, diagnosis, or treatment. Always consult with qualified 
              healthcare providers for medical decisions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Navigation />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/predict" element={<DiseaseSelection />} />
          <Route path="/predict/diabetes" element={<DiabetesForm />} />
          <Route path="/predict/heart" element={<HeartDiseaseForm />} />
          <Route path="/predict/parkinsons" element={<ParkinsonsForm />} />
          <Route path="/history" element={<History />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;