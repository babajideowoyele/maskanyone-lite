import React, { useEffect, useState } from 'react';
import {
  Button,
  FileUploaderDropContainer,
  FileUploaderItem,
  Header,
  HeaderName,
  InlineLoading,
  RadioButton,
  RadioButtonGroup,
  Select,
  SelectItem,
  Slider,
  Tag,
  Theme,
} from '@carbon/react';

const STRATEGIES = ['blur', 'solid', 'pixelate'];
const MODES = ['quick', 'precision'];

export default function App() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState('quick');
  const [strategy, setStrategy] = useState('blur');
  const [downsample, setDownsample] = useState(1.0);
  const [job, setJob] = useState(null); // {job_id, status, error, ...}
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!job || job.status === 'done' || job.status === 'failed') return;
    const t = setInterval(async () => {
      const r = await fetch(`/mask/${job.job_id}`);
      if (r.ok) setJob(await r.json());
    }, 2000);
    return () => clearInterval(t);
  }, [job]);

  async function submit() {
    if (!file) return;
    setSubmitting(true);
    try {
      const fd = new FormData();
      fd.append('video', file);
      fd.append('mode', mode);
      fd.append('strategy', strategy);
      fd.append('downsample', String(downsample));
      const r = await fetch('/mask', { method: 'POST', body: fd });
      if (!r.ok) throw new Error(`upload failed: ${r.status}`);
      setJob(await r.json());
    } catch (e) {
      setJob({ status: 'failed', error: e.message });
    } finally {
      setSubmitting(false);
    }
  }

  function reset() {
    setJob(null);
    setFile(null);
  }

  const running = job && (job.status === 'pending' || job.status === 'running');
  const done = job && job.status === 'done';
  const failed = job && job.status === 'failed';

  return (
    <Theme theme="white">
      <Header aria-label="maskanyone-lite">
        <HeaderName prefix="maskanyone">lite</HeaderName>
      </Header>

      <main className="content stack">
        <div className="stack">
          <h1>Privacy-aware video masking</h1>
          <p className="lede">
            Upload a video; pick a mode and strategy. Quick uses a lightweight
            person silhouette; precision uses SAM-compatible segmentation. Both
            run on CPU. Every output ships with a manifest recording the exact
            model, parameters, and input checksum — cite-ready.
          </p>
        </div>

        {!job && (
          <div className="stack">
            <FileUploaderDropContainer
              labelText="Drop a video file here or click to select"
              accept={['video/mp4', 'video/quicktime', 'video/x-matroska']}
              onAddFiles={(_e, { addedFiles }) => setFile(addedFiles[0] || null)}
            />
            {file && (
              <FileUploaderItem
                name={file.name}
                status="edit"
                onDelete={() => setFile(null)}
              />
            )}

            <RadioButtonGroup
              legendText="Mode"
              name="mode"
              valueSelected={mode}
              onChange={setMode}
            >
              {MODES.map((m) => (
                <RadioButton key={m} id={`mode-${m}`} labelText={m} value={m} />
              ))}
            </RadioButtonGroup>

            <div className="row">
              <Select
                id="strategy"
                labelText="Strategy"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
              >
                {STRATEGIES.map((s) => (
                  <SelectItem key={s} text={s} value={s} />
                ))}
              </Select>
              <Slider
                labelText={`Downsample (${downsample.toFixed(2)})`}
                min={0.1}
                max={1.0}
                step={0.05}
                value={downsample}
                onChange={({ value }) => setDownsample(value)}
              />
            </div>

            <p className="muted">
              Lower downsample = faster precision mode. 1.0 = native resolution.
            </p>

            <div className="actions">
              <Button
                disabled={!file || submitting}
                onClick={submit}
                renderIcon={submitting ? undefined : undefined}
              >
                {submitting ? 'uploading…' : 'Mask video'}
              </Button>
            </div>
          </div>
        )}

        {job && (
          <div className="stack">
            <div className="actions">
              <Tag type={done ? 'green' : failed ? 'red' : 'gray'}>
                {job.status}
              </Tag>
              <span className="muted">job {job.job_id?.slice(0, 8)}</span>
            </div>

            {running && (
              <InlineLoading description="masking — safe to leave this tab" />
            )}

            {done && (
              <div className="actions">
                <Button
                  as="a"
                  href={`/mask/${job.job_id}/result`}
                  download
                >
                  Download zip
                </Button>
                <Button kind="ghost" onClick={reset}>
                  Mask another
                </Button>
              </div>
            )}

            {failed && (
              <div className="stack">
                <p>Job failed.</p>
                {job.error && (
                  <pre className="muted" style={{ whiteSpace: 'pre-wrap' }}>
                    {job.error}
                  </pre>
                )}
                <Button kind="ghost" onClick={reset}>
                  Try again
                </Button>
              </div>
            )}

            <div className="muted">
              <div>mode: {job.mode}</div>
              <div>strategy: {job.strategy}</div>
              <div>original filename: {job.original_filename}</div>
              {job.started_at && <div>started: {job.started_at}</div>}
              {job.finished_at && <div>finished: {job.finished_at}</div>}
            </div>
          </div>
        )}
      </main>
    </Theme>
  );
}
