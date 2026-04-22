--
-- PostgreSQL database dump
--

-- Dumped from database version 17.4
-- Dumped by pg_dump version 17.5 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA public;


--
-- Name: actor_type; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.actor_type AS ENUM (
    'user',
    'job_manager',
    'job_processor',
    'worker_process',
    'system'
);


--
-- Name: TYPE actor_type; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TYPE public.actor_type IS 'Attribution source of job_events. Indicates which actor type triggered the event.';


--
-- Name: asset_upload_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.asset_upload_status AS ENUM (
    'pending',
    'processing',
    'ready',
    'invalid_mime',
    'error'
);


--
-- Name: job_event_action; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.job_event_action AS ENUM (
    'job_queued',
    'job_dequeued',
    'attempt_started',
    'job_succeeded',
    'attempt_failed',
    'attempt_retry_scheduled',
    'attempt_retry_exhausted',
    'job_marked_dead',
    'manual_attempt_started',
    'job_enqueue_failed',
    'log_info',
    'log_warning',
    'log_exception'
);


--
-- Name: TYPE job_event_action; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TYPE public.job_event_action IS 'Describes lifecycle and diagnostic events for jobs. Used in job_events.';


--
-- Name: job_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.job_status AS ENUM (
    'queued',
    'dequeued',
    'processing',
    'done',
    'error',
    'dead',
    'enqueue_failed'
);


--
-- Name: TYPE job_status; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TYPE public.job_status IS 'Used in jobs table: now extended with value `dead` to indicate terminal failed state.';


--
-- Name: photobook_status; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.photobook_status AS ENUM (
    'draft',
    'pending',
    'deleted',
    'permanently_deleted',
    'published',
    'uploading',
    'upload_failed',
    'ready_for_generation',
    'generating',
    'generation_failed'
);


--
-- Name: photobook_status_editor; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.photobook_status_editor AS ENUM (
    'user',
    'upload_pipeline',
    'generation_pipeline',
    'system'
);


--
-- Name: TYPE photobook_status_editor; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TYPE public.photobook_status_editor IS 'Actor that triggered photobook status update.';


--
-- Name: user_provided_occasion; Type: TYPE; Schema: public; Owner: -
--

CREATE TYPE public.user_provided_occasion AS ENUM (
    'wedding',
    'birthday',
    'anniversary',
    'other'
);


--
-- Name: handle_user_delete(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.handle_user_delete() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
BEGIN
    DELETE FROM public.users WHERE id = OLD.id;
    RETURN OLD;
END;
$$;


--
-- Name: handle_user_insert(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.handle_user_insert() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
    user_name TEXT := (NEW.raw_user_meta_data->>'name');
BEGIN
    INSERT INTO public.users (
        id,
        email,
        phone,
        email_confirmed_at,
        phone_confirmed_at,
        name
    )
    VALUES (
        NEW.id,
        NEW.email,
        NEW.phone,
        NEW.email_confirmed_at,
        NEW.phone_confirmed_at,
        user_name
    );
    RETURN NEW;
END;
$$;


--
-- Name: handle_user_update(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.handle_user_update() RETURNS trigger
    LANGUAGE plpgsql SECURITY DEFINER
    AS $$
DECLARE
    user_name TEXT := (NEW.raw_user_meta_data->>'name');
BEGIN
    UPDATE public.users
    SET
        email = NEW.email,
        phone = NEW.phone,
        email_confirmed_at = NEW.email_confirmed_at,
        phone_confirmed_at = NEW.phone_confirmed_at,
        name = user_name
    WHERE id = NEW.id;
    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: assets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assets (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    asset_key_original text,
    asset_key_display text,
    asset_key_llm text,
    metadata_json jsonb,
    created_at timestamp with time zone DEFAULT now(),
    original_photobook_id uuid,
    upload_status public.asset_upload_status DEFAULT 'pending'::public.asset_upload_status NOT NULL
);


--
-- Name: job_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.job_events (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    job_id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    event_action public.job_event_action NOT NULL,
    message text,
    host text,
    retry_count integer DEFAULT 0,
    extra jsonb,
    actor_type public.actor_type NOT NULL,
    actor_id text,
    job_type text,
    photobook_id uuid
);


--
-- Name: TABLE job_events; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.job_events IS 'Immutable audit trail for job transitions, retries, errors, and debug logs.';


--
-- Name: COLUMN job_events.event_action; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.job_events.event_action IS 'Event type that occurred (e.g. retry_scheduled, log_warning).';


--
-- Name: COLUMN job_events.actor_type; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.job_events.actor_type IS 'Who performed the action: system, worker, user, etc.';


--
-- Name: COLUMN job_events.actor_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.job_events.actor_id IS 'ID of actor (e.g. user UUID, system ID, worker ID) if available.';


--
-- Name: jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.jobs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    job_type text NOT NULL,
    status public.job_status DEFAULT 'queued'::public.job_status NOT NULL,
    input_payload jsonb,
    result_payload jsonb,
    error_message text,
    user_id uuid,
    photobook_id uuid,
    created_at timestamp with time zone DEFAULT now(),
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    retry_count integer DEFAULT 0,
    max_retries integer DEFAULT 3,
    last_attempted_at timestamp with time zone
);


--
-- Name: COLUMN jobs.retry_count; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.jobs.retry_count IS 'Number of retry attempts this job has made.';


--
-- Name: COLUMN jobs.max_retries; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.jobs.max_retries IS 'Maximum retries allowed before job is marked dead.';


--
-- Name: COLUMN jobs.last_attempted_at; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.jobs.last_attempted_at IS 'Timestamp of the last job execution attempt.';


--
-- Name: pages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pages (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    photobook_id uuid,
    page_number integer NOT NULL,
    user_message text,
    layout text,
    created_at timestamp with time zone DEFAULT now(),
    user_message_alternative_options jsonb
);


--
-- Name: pages_assets_rel; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pages_assets_rel (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    page_id uuid,
    asset_id uuid,
    order_index integer,
    caption text
);


--
-- Name: photobook_bookmarks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.photobook_bookmarks (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    photobook_id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    source text
);


--
-- Name: photobooks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.photobooks (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    title text NOT NULL,
    caption text,
    theme text,
    status public.photobook_status DEFAULT 'draft'::public.photobook_status,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    user_provided_occasion public.user_provided_occasion,
    user_provided_occasion_custom_details text,
    user_provided_context text,
    thumbnail_asset_id uuid,
    deleted_at timestamp with time zone,
    status_last_edited_by public.photobook_status_editor DEFAULT 'user'::public.photobook_status_editor
);


--
-- Name: COLUMN photobooks.status_last_edited_by; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.photobooks.status_last_edited_by IS 'Indicates which component last updated status.';


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.schema_migrations (
    version character varying NOT NULL
);


--
-- Name: users; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.users (
    id uuid NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    email text,
    phone text,
    email_confirmed_at timestamp with time zone,
    phone_confirmed_at timestamp with time zone,
    name text,
    role text DEFAULT 'user'::text NOT NULL
);


--
-- Name: assets assets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assets
    ADD CONSTRAINT assets_pkey PRIMARY KEY (id);


--
-- Name: job_events job_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_events
    ADD CONSTRAINT job_events_pkey PRIMARY KEY (id);


--
-- Name: jobs jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_pkey PRIMARY KEY (id);


--
-- Name: pages_assets_rel pages_assets_rel_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pages_assets_rel
    ADD CONSTRAINT pages_assets_rel_pkey PRIMARY KEY (id);


--
-- Name: pages pages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pages
    ADD CONSTRAINT pages_pkey PRIMARY KEY (id);


--
-- Name: photobook_bookmarks photobook_bookmarks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.photobook_bookmarks
    ADD CONSTRAINT photobook_bookmarks_pkey PRIMARY KEY (id);


--
-- Name: photobooks photobooks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.photobooks
    ADD CONSTRAINT photobooks_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: photobook_bookmarks unique_user_photobook; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.photobook_bookmarks
    ADD CONSTRAINT unique_user_photobook UNIQUE (user_id, photobook_id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: idx_job_events_job_id_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_job_events_job_id_created_at ON public.job_events USING btree (job_id, created_at);


--
-- Name: idx_job_events_photobook_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_job_events_photobook_id ON public.job_events USING btree (photobook_id);


--
-- Name: idx_jobs_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_created_at ON public.jobs USING btree (created_at);


--
-- Name: idx_jobs_job_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_job_type ON public.jobs USING btree (job_type);


--
-- Name: idx_jobs_last_attempted_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_last_attempted_at ON public.jobs USING btree (last_attempted_at);


--
-- Name: idx_jobs_photobook_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_photobook_id ON public.jobs USING btree (photobook_id);


--
-- Name: idx_jobs_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_status ON public.jobs USING btree (status);


--
-- Name: idx_jobs_status_job_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_status_job_type ON public.jobs USING btree (status, job_type);


--
-- Name: idx_jobs_status_retry_count; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_status_retry_count ON public.jobs USING btree (status, retry_count);


--
-- Name: idx_pages_assets_rel_asset_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pages_assets_rel_asset_id ON public.pages_assets_rel USING btree (asset_id);


--
-- Name: idx_pages_assets_rel_page_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pages_assets_rel_page_id ON public.pages_assets_rel USING btree (page_id);


--
-- Name: idx_pages_photobook_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pages_photobook_id ON public.pages USING btree (photobook_id);


--
-- Name: idx_photobook_bookmarks_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_photobook_bookmarks_user_id ON public.photobook_bookmarks USING btree (user_id);


--
-- Name: idx_photobooks_thumbnail_asset_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_photobooks_thumbnail_asset_id ON public.photobooks USING btree (thumbnail_asset_id);


--
-- Name: assets assets_original_photobook_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assets
    ADD CONSTRAINT assets_original_photobook_id_fkey FOREIGN KEY (original_photobook_id) REFERENCES public.photobooks(id);


--
-- Name: photobook_bookmarks fk_bookmark_photobook; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.photobook_bookmarks
    ADD CONSTRAINT fk_bookmark_photobook FOREIGN KEY (photobook_id) REFERENCES public.photobooks(id) ON DELETE CASCADE;


--
-- Name: photobook_bookmarks fk_bookmark_user; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.photobook_bookmarks
    ADD CONSTRAINT fk_bookmark_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE;


--
-- Name: job_events job_events_job_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_events
    ADD CONSTRAINT job_events_job_id_fkey FOREIGN KEY (job_id) REFERENCES public.jobs(id) ON DELETE CASCADE;


--
-- Name: job_events job_events_photobook_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.job_events
    ADD CONSTRAINT job_events_photobook_id_fkey FOREIGN KEY (photobook_id) REFERENCES public.photobooks(id) ON DELETE SET NULL;


--
-- Name: jobs jobs_photobook_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_photobook_id_fkey FOREIGN KEY (photobook_id) REFERENCES public.photobooks(id);


--
-- Name: pages_assets_rel pages_assets_rel_asset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pages_assets_rel
    ADD CONSTRAINT pages_assets_rel_asset_id_fkey FOREIGN KEY (asset_id) REFERENCES public.assets(id);


--
-- Name: pages_assets_rel pages_assets_rel_page_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pages_assets_rel
    ADD CONSTRAINT pages_assets_rel_page_id_fkey FOREIGN KEY (page_id) REFERENCES public.pages(id);


--
-- Name: pages pages_photobook_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pages
    ADD CONSTRAINT pages_photobook_id_fkey FOREIGN KEY (photobook_id) REFERENCES public.photobooks(id);


--
-- Name: photobooks photobooks_thumbnail_asset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.photobooks
    ADD CONSTRAINT photobooks_thumbnail_asset_id_fkey FOREIGN KEY (thumbnail_asset_id) REFERENCES public.assets(id) ON DELETE SET NULL;


--
-- Name: users users_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

