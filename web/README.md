# FPL Intelligence React Frontend

React/Vite frontend for the FPL Intelligence Platform.

The Python ETL/model pipeline remains the source of truth:

```text
FPL API -> Python ETL/model -> Supabase -> React frontend
```

## Local setup

Create `web/.env`:

```env
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

Then run:

```bash
npm install
npm run dev
```

## Deploy

Deploy the `web/` directory to Vercel or Netlify.

Build command:

```bash
npm run build
```

Output directory:

```text
dist
```

Set the same `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` variables in the hosting provider.
