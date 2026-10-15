import { ChangeEvent, useState } from 'react';
import { AlertCircle, Download, Play, RotateCcw, Sparkles, Upload } from 'lucide-react';
import { api, Operation } from './api';

type Profile = {
  metrics: { rows: number; columns: number; missing_cells: number; duplicate_rows: number };
  columns: { name: string; dtype: string; unique: number; missing: number; suspicious?: string }[];
  range_violations?: { column: string; invalid_count: number; min: number; max: number; rule: string }[];
};
type Preview = { columns: string[]; rows: Record<string, unknown>[] };
type CleaningResult = { before: { completeness: number }; after: { completeness: number; record_retention: number } };
type Candidate = { candidate_id: string; operation: string; strategy: string; score: number; metrics: { rows: number }; risk: string };
type Simulation = { formula: string; candidates: Candidate[]; comparison_notes?: string[] };
type View = 'dashboard' | 'profile' | 'history';
type HistoryEntry = {
  id: number; created_at: string; operation: string; parameters: string; status: string; reason: string;
  affected_rows: number; affected_cells: number;
};
type IntentOperation = { name: Operation['name']; columns: string[]; confidence: string; reason: string; similarity?: number; matched_example?: string; column_reason?: string; negated?: boolean };
type ObjectiveOperation = Operation & { confidence: string; source: 'objective'; note: string; similarity?: number; matched_example?: string; column_reason?: string };
type PlanResponse = {
  intent: { operations: IntentOperation[]; unsupported: string[]; assumptions: string[]; warnings?: string[]; normalized_instruction?: string };
  objective_operations: ObjectiveOperation[];
  profile_recommendations: { name: string; reason: string }[];
  manual_operations: Operation[];
  validation_errors: string[];
};

const OPS: Operation[] = [
  { name: 'missing', params: { strategy: 'median' }, reason: 'Handle missing values' },
  { name: 'duplicates', reason: 'Remove exact duplicate rows' },
  { name: 'outliers', params: { strategy: 'flag', method: 'iqr' }, reason: 'Flag statistical outliers' },
  { name: 'categorical', params: { case_normalize: false }, reason: 'Trim categorical values' },
  { name: 'scale', params: { strategy: 'standard' }, reason: 'Scale numeric features' },
];

export default function App() {
  const [sid, setSid] = useState('');
  const [profile, setProfile] = useState<Profile | null>(null);
  const [objective, setObjective] = useState('');
  const [selected, setSelected] = useState<Operation[]>([]);
  const [message, setMessage] = useState('Upload a dataset to begin.');
  const [preview, setPreview] = useState<Preview | null>(null);
  const [result, setResult] = useState<CleaningResult | null>(null);
  const [sim, setSim] = useState<Simulation | null>(null);
  const [planResult, setPlanResult] = useState<PlanResponse | null>(null);
  const [planning, setPlanning] = useState(false);
  const [objectiveAdded, setObjectiveAdded] = useState<Set<Operation['name']>>(new Set());
  const [planApproved, setPlanApproved] = useState(false);
  const [view, setView] = useState<View>('dashboard');
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState('');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState('');

  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      setMessage('Uploading and profiling…');
      const body = new FormData();
      body.append('file', file);
      const response = await api.post('/datasets/upload', body);
      setSid(response.data.session_id);
      setProfile(response.data.profile);
      const previewResponse = await api.get('/datasets/' + response.data.session_id + '/preview');
      setPreview(previewResponse.data);
      setMessage(file.name + ' is ready for review.');
      setPlanResult(null);
      setObjectiveAdded(new Set());
      setPlanApproved(false);
      setHistory([]);
      setView('dashboard');
    } catch (error: any) {
      setMessage(error.response?.data?.detail || 'Upload failed.');
    }
  };

  const plan = async () => {
    if (!sid) return;
    setPlanning(true);
    try {
      const response = await api.post('/datasets/' + sid + '/plan', { objective, selected_operations: selected });
      setPlanResult(response.data as PlanResponse);
      setPlanApproved(false);
      setMessage('Objective interpreted. Review the transparent recommendations below before adding them.');
    } catch (error: any) {
      setPlanResult(null);
      setMessage(error.response?.data?.detail || 'Plan could not be generated.');
    } finally {
      setPlanning(false);
    }
  };

  const refreshDataset = async () => {
    const [previewResponse, profileResponse] = await Promise.all([
      api.get('/datasets/' + sid + '/preview'),
      api.get('/datasets/' + sid + '/profile'),
    ]);
    setPreview(previewResponse.data);
    setProfile(profileResponse.data);
  };

  const loadProfile = async () => {
    if (!sid) return;
    setProfileLoading(true);
    setProfileError('');
    try {
      await refreshDataset();
    } catch (error: any) {
      setProfileError(error.response?.data?.detail || 'Could not refresh the dataset profile.');
    } finally {
      setProfileLoading(false);
    }
  };

  const loadHistory = async () => {
    if (!sid) return;
    setHistoryLoading(true);
    setHistoryError('');
    try {
      const response = await api.get('/datasets/' + sid + '/history');
      setHistory(response.data);
    } catch (error: any) {
      setHistoryError(error.response?.data?.detail || 'Could not load cleaning history.');
    } finally {
      setHistoryLoading(false);
    }
  };

  const navigate = async (nextView: View) => {
    setView(nextView);
    if (nextView === 'profile') await loadProfile();
    if (nextView === 'history') await loadHistory();
  };

  const clean = async () => {
    if (!planApproved) {
      setMessage('Review the plan and confirm approval before running cleaning.');
      return;
    }
    try {
      const approvedOperations = selected.map((operation) => operation.name === 'outliers'
        && operation.params?.strategy === 'remove'
        ? { ...operation, params: { ...operation.params, approved: true } }
        : operation);
      const response = await api.post('/datasets/' + sid + '/clean', { operations: approvedOperations, confirm_destructive: true });
      setResult(response.data);
      await refreshDataset();
      setMessage('Cleaning completed. Original upload is preserved.');
    } catch (error: any) {
      setMessage(error.response?.data?.detail || 'Cleaning could not run.');
    }
  };

  const simulate = async () => {
    try {
      const response = await api.post('/datasets/' + sid + '/simulate', { operations: selected });
      setSim(response.data);
      setMessage('Simulation compared isolated copies; nothing has been changed.');
    } catch (error: any) {
      setMessage(error.response?.data?.detail || 'No useful simulation candidates for this plan.');
    }
  };

  const undoLatest = async () => {
    try {
      const historyResponse = await api.get('/datasets/' + sid + '/history');
      const latest = historyResponse.data[0];
      if (!latest) {
        setMessage('There is no operation to undo.');
        return;
      }
      await api.post('/datasets/' + sid + '/undo/' + latest.id);
      await refreshDataset();
      setMessage('Restored latest saved snapshot.');
    } catch (error: any) {
      setMessage(error.response?.data?.detail || 'Could not restore the snapshot.');
    }
  };

  const undoHistoryEntry = async (historyId: number) => {
    try {
      await api.post('/datasets/' + sid + '/undo/' + historyId);
      await refreshDataset();
      await loadHistory();
      setResult(null);
      setMessage('Restored the selected saved snapshot.');
    } catch (error: any) {
      setMessage(error.response?.data?.detail || 'Could not restore the selected snapshot.');
    }
  };

  const toggle = (operation: Operation) => {
    setObjectiveAdded((current) => {
      const next = new Set(current);
      next.delete(operation.name);
      return next;
    });
    setSelected((current) => current.some((item) => item.name === operation.name)
      ? current.filter((item) => item.name !== operation.name)
      : [...current, operation]);
  };

  const updateOperation = (name: Operation['name'], update: Partial<Operation>) => {
    setSelected((current) => current.map((operation) => operation.name === name
      ? { ...operation, ...update, params: { ...operation.params, ...update.params } }
      : operation));
    setPlanApproved(false);
  };

  const addObjectiveRecommendations = () => {
    if (!planResult) return;
    const additions = planResult.objective_operations.filter(
      (recommendation) => !selected.some((operation) => operation.name === recommendation.name),
    );
    setSelected((current) => [...current, ...additions]);
    setObjectiveAdded((current) => new Set([...current, ...additions.map((operation) => operation.name)]));
    setMessage(additions.length
      ? 'Objective recommendations were added. You can still modify every operation before cleaning.'
      : 'All objective recommendations are already selected.');
    setPlanApproved(false);
  };

  return (
    <main>
      <aside>
        <div className="brand"><Sparkles /> DataCleanAI</div>
        <p>Rule-based data profiling and safe preprocessing.</p>
        <nav>
          <button className={view === 'dashboard' ? 'active' : ''} onClick={() => navigate('dashboard')}>Dashboard</button>
          <button className={view === 'profile' ? 'active' : ''} onClick={() => navigate('profile')}>Dataset profile</button>
          <button className={view === 'history' ? 'active' : ''} onClick={() => navigate('history')}>Cleaning history</button>
        </nav>
        <small>Local-first · No external AI</small>
      </aside>
      <section className="content">
        <header>
          <div>
            <h1>Data cleaning workspace</h1>
            <p>Profile, review a transparent plan, then choose quick cleaning or simulation.</p>
          </div>
          {sid && <span className="pill">Session {sid.slice(0, 8)}</span>}
        </header>
        <div className="notice"><AlertCircle size={18} />{message}</div>

        {view === 'dashboard' ? <><div className="grid">
          <article className="card upload">
            <label htmlFor="file">
              <Upload />
              <strong>Upload a dataset</strong>
              <span>CSV, Excel, or JSON records · max 20 MB</span>
            </label>
            <input id="file" type="file" accept=".csv,.xlsx,.xls,.json" onChange={upload} />
          </article>
          {profile && (
            <article className="card stats">
              <b>{profile.metrics.rows}</b><span>rows</span>
              <b>{profile.metrics.columns}</b><span>columns</span>
              <b>{profile.metrics.missing_cells}</b><span>missing cells</span>
              <b>{profile.metrics.duplicate_rows}</b><span>duplicates</span>
            </article>
          )}
        </div>

        {profile && (
          <>
            <article className="card">
              <h2>Objective</h2>
              <textarea
                aria-label="Cleaning objective"
                value={objective}
                onChange={(event) => setObjective(event.target.value)}
                placeholder="e.g. Remove duplicate records, fill missing marks, and identify outliers."
              />
              <button onClick={plan} disabled={planning}>
                <Sparkles size={16} /> {planning ? 'Interpreting…' : 'Interpret and recommend'}
              </button>
            </article>
            {planResult && (
              <article className="card plan-review">
                <h2>Objective interpretation</h2>
                {!objective.trim() && <p>No objective was entered. No objective-based operations were recommended.</p>}
                {planResult.intent.operations.length > 0 && (
                  <div className="plan-list">
                    {planResult.intent.operations.map((item) => (
                      <div key={item.name}>
                        <strong>{item.name}</strong> <span className="pill">{item.confidence}</span>
                        <p>{item.reason} {item.columns.length ? 'Matched columns: ' + item.columns.join(', ') + '.' : 'No column was matched.'}</p>
                        <p className="muted">Lexical cosine similarity: {item.similarity ?? 0} (not a probability); closest catalog example: “{item.matched_example}”. {item.column_reason}</p>
                      </div>
                    ))}
                  </div>
                )}
                {planResult.objective_operations.length > 0 && (
                  <>
                    <h3>Recommended from your objective</h3>
                    {planResult.objective_operations.map((item) => (
                      <div className="recommendation" key={item.name}>
                        <div><strong>{item.name}</strong><p>{item.reason} {item.note}</p><p className="muted">{item.column_reason}</p></div>
                      </div>
                    ))}
                    <button onClick={addObjectiveRecommendations}>Add objective recommendations to my selection</button>
                  </>
                )}
                {planResult.profile_recommendations.length > 0 && (
                  <p className="muted">Dataset profile signals: {planResult.profile_recommendations.map((item) => item.name).join(', ')}. These are informational and are not selected automatically.</p>
                )}
                {planResult.intent.assumptions.map((item) => <p className="muted" key={item}>Assumption: {item}</p>)}
                {(planResult.intent.warnings || []).map((item) => <p className="warning" key={item}>{item}</p>)}
                {planResult.intent.unsupported.map((item) => <p className="warning" key={item}>{item}</p>)}
                {planResult.validation_errors.map((item) => <p className="warning" key={item}>{item}</p>)}
              </article>
            )}
            <div className="columns">
              <article className="card">
                <h2>Operations</h2>
                {OPS.map((operation) => (
                  <label className="check" key={operation.name}>
                    <input
                      type="checkbox"
                      checked={selected.some((item) => item.name === operation.name)}
                      onChange={() => toggle(operation)}
                    />
                    <span>
                      <strong>{operation.reason}</strong>
                      {selected.some((item) => item.name === operation.name) && (
                        <small className="source-label">
                          {objectiveAdded.has(operation.name) ? 'Added from objective recommendation' : 'Manual selection'}
                        </small>
                      )}
                      <small>
                        {operation.name === 'scale'
                          ? 'For downstream modelling; opt-in.'
                          : operation.name === 'outliers'
                            ? 'Flags only by default; selected flagging adds an _outlier_flag output column.'
                            : 'Review settings on server defaults.'}
                      </small>
                      {selected.some((item) => item.name === operation.name) && (
                        <span className="operation-editor">
                          <label>
                            Columns
                            <select
                              value={(selected.find((item) => item.name === operation.name)?.columns || []).join(',')}
                              onChange={(event) => updateOperation(operation.name, {
                                columns: event.target.value ? event.target.value.split(',') : [],
                              })}
                            >
                              <option value="">Automatic / full-row</option>
                              {profile.columns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
                            </select>
                          </label>
                          {operation.name === 'missing' && (
                            <label>Strategy
                              <select value={String(selected.find((item) => item.name === operation.name)?.params?.strategy || 'median')}
                                onChange={(event) => updateOperation(operation.name, { params: { strategy: event.target.value } })}>
                                <option value="median">Median</option><option value="mean">Mean</option><option value="mode">Mode</option><option value="drop_rows">Drop rows</option><option value="leave">Leave unchanged</option>
                              </select>
                            </label>
                          )}
                          {operation.name === 'outliers' && (
                            <label>Strategy
                              <select value={String(selected.find((item) => item.name === operation.name)?.params?.strategy || 'flag')}
                                onChange={(event) => updateOperation(operation.name, { params: { strategy: event.target.value } })}>
                                <option value="flag">Flag only</option><option value="cap">Cap extremes</option><option value="remove">Remove (requires review)</option>
                              </select>
                            </label>
                          )}
                        </span>
                      )}
                    </span>
                  </label>
                ))}
                <div className="actions">
                  <label className="approval"><input type="checkbox" checked={planApproved} onChange={(event) => setPlanApproved(event.target.checked)} /> I approve this reviewed plan</label>
                  <button className="primary" disabled={!selected.length || !planApproved} onClick={clean}>
                    <Play size={16} /> Quick clean
                  </button>
                  <button disabled={!selected.length} onClick={simulate}>Compare strategies</button>
                </div>
              </article>
              <article className="card">
                <h2>Dataset preview</h2>
                <div className="table">
                  <table>
                    <thead><tr>{preview?.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
                    <tbody>
                      {preview?.rows.map((row, index) => (
                        <tr key={index}>
                          {preview.columns.map((column) => <td key={column}>{String(row[column] ?? '')}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <h3>Column profile</h3>
                {profile.columns.map((column) => (
                  <div className="column" key={column.name}>
                    <b>{column.name}</b>
                    <span>
                      {column.dtype} · {column.unique} unique · {column.missing} missing
                      {column.suspicious && (' · ' + column.suspicious)}
                    </span>
                  </div>
                ))}
                {(profile.range_violations || []).map((finding) => (
                  <p className="warning" key={finding.column}>
                    Range validation: {finding.column} has {finding.invalid_count} value(s) outside 0–100 ({finding.min} to {finding.max}). No values were changed.
                  </p>
                ))}
              </article>
            </div>
          </>
        )}

        {result && (
          <article className="card result">
            <h2>Before / after</h2>
            <p>Completeness: {result.before.completeness} → {result.after.completeness} · Retention: {result.after.record_retention}</p>
            <div className="actions">
              <a href={api.defaults.baseURL + '/datasets/' + sid + '/export/csv'}><Download size={16} /> Cleaned CSV</a>
              <a href={api.defaults.baseURL + '/datasets/' + sid + '/export/report'}><Download size={16} /> JSON report</a>
              <button onClick={undoLatest}><RotateCcw size={16} /> Undo latest</button>
            </div>
          </article>
        )}

        {sim && (
          <article className="card">
            <h2>Strategy simulator (advisory)</h2>
            <p>{sim.formula}</p>
            {(sim.comparison_notes || []).map((note) => <p className="warning" key={note}>{note}</p>)}
            <div className="table">
              <table>
                <thead><tr><th>Operation</th><th>Strategy</th><th>Score</th><th>Rows</th><th>Risk</th></tr></thead>
                <tbody>
                  {sim.candidates.map((candidate) => (
                    <tr key={candidate.candidate_id}>
                      <td>{candidate.operation}</td>
                      <td>{candidate.strategy}</td>
                      <td>{candidate.score}</td>
                      <td>{candidate.metrics.rows}</td>
                      <td>{candidate.risk}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        )}</> : view === 'profile' ? (
          <section className="view-page">
            <h2>Dataset profile</h2>
            {!sid && <div className="card empty-state">Upload a dataset to view its real profile.</div>}
            {sid && profileLoading && <div className="card">Refreshing profile…</div>}
            {sid && profileError && <div className="card warning">{profileError}</div>}
            {sid && profile && !profileLoading && (
              <>
                <div className="card stats profile-stats">
                  <b>{profile.metrics.rows}</b><span>rows</span><b>{profile.metrics.columns}</b><span>columns</span>
                  <b>{profile.metrics.missing_cells}</b><span>missing cells</span><b>{profile.metrics.duplicate_rows}</b><span>duplicate rows</span>
                </div>
                <article className="card">
                  <h3>Columns and quality indicators</h3>
                  <div className="table"><table><thead><tr><th>Column</th><th>Type</th><th>Unique</th><th>Missing</th><th>Indicator</th></tr></thead>
                    <tbody>{profile.columns.map((column) => <tr key={column.name}><td>{column.name}</td><td>{column.dtype}</td><td>{column.unique}</td><td>{column.missing}</td><td>{column.suspicious || '—'}</td></tr>)}</tbody>
                  </table></div>
                  {(profile.range_violations || []).map((finding) => <p className="warning" key={finding.column}>Range validation: {finding.rule} {finding.column}: {finding.invalid_count} violation(s).</p>)}
                </article>
                <article className="card"><h3>Current preview</h3><div className="table"><table><thead><tr>{preview?.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{preview?.rows.map((row, index) => <tr key={index}>{preview.columns.map((column) => <td key={column}>{String(row[column] ?? '')}</td>)}</tr>)}</tbody></table></div></article>
              </>
            )}
          </section>
        ) : (
          <section className="view-page">
            <h2>Cleaning history</h2>
            {!sid && <div className="card empty-state">Upload a dataset to view its cleaning history.</div>}
            {sid && historyLoading && <div className="card">Loading persisted history…</div>}
            {sid && historyError && <div className="card warning">{historyError}</div>}
            {sid && !historyLoading && !historyError && history.length === 0 && <div className="card empty-state">No cleaning operations have been completed for this dataset.</div>}
            {sid && !historyLoading && history.map((entry) => (
              <article className="card history-entry" key={entry.id}>
                <div><h3>{entry.operation}</h3><p>{entry.reason || 'No reason recorded.'}</p><small>{new Date(entry.created_at).toLocaleString()} · {entry.status}</small></div>
                <p>{entry.affected_cells} changed cells · {entry.affected_rows} affected rows</p>
                <details><summary>Parameters</summary><pre>{entry.parameters}</pre></details>
                <button onClick={() => undoHistoryEntry(entry.id)}><RotateCcw size={16} /> Restore this snapshot</button>
              </article>
            ))}
          </section>
        )}
      </section>
    </main>
  );
}
