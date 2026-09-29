-- Tablas de LSM Coach en Supabase.
-- Dónde: Supabase > tu proyecto > SQL Editor > New query > pegar todo > Run.

create table if not exists public.samples (
  id          uuid primary key,
  created_at  timestamptz not null default now(),
  person      text not null,
  sign        text not null,                    -- 'A', 'B', ... o la palabra
  is_error    boolean not null default false,   -- true = error intencional (para medir aciertos)
  source      text not null default 'propio',   -- 'propio' o el dataset público de origen
  camera      text,
  angles      jsonb not null,                   -- {"pulgar": grados, "indice": ..., ...}
  landmarks   jsonb,                            -- 63 números normalizados (21 puntos x, y, z)
  imu         jsonb                             -- {"roll", "pitch", "yaw"} o null
);
create index if not exists samples_sign_idx on public.samples (sign, is_error);

-- Registro de progreso del aprendiz: una fila cada vez que logra una seña
create table if not exists public.attempts (
  id                 uuid primary key,
  created_at         timestamptz not null default now(),
  person             text not null,
  sign               text not null,
  seconds            real,                      -- tiempo desde que eligió la seña hasta lograrla
  failed_parameters  jsonb                      -- parámetros que falló antes: ["Configuración", ...]
);
create index if not exists attempts_person_idx on public.attempts (person, sign);

-- Seguridad (funciona con "Automatically expose new tables" activado o desactivado):
--  * solo la clave SECRETA (rol service_role) puede usar las tablas por la Data API;
--  * la clave pública (anon / publishable) no tiene ningún permiso;
--  * RLS activado y SIN políticas, como segunda barrera.
-- La clave secreta vive solo en .env de la laptop / UNO Q, nunca en la página web.
grant usage on schema public to service_role;
grant select, insert, update on public.samples, public.attempts to service_role;
revoke all on public.samples, public.attempts from anon, authenticated;

alter table public.samples enable row level security;
alter table public.attempts enable row level security;
