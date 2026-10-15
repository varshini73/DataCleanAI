import axios from 'axios';
export const api=axios.create({baseURL:import.meta.env.VITE_API_URL||'http://127.0.0.1:8000/api'});
export type Operation={name:'missing'|'duplicates'|'outliers'|'convert'|'categorical'|'scale';enabled?:boolean;columns?:string[];params?:Record<string,unknown>;reason?:string};
