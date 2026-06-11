use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use uuid::Uuid;

use crate::AppState;

// ── Session store ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AuthConfig {
    username: String,
    password: String,
    sessions: Arc<Mutex<HashMap<String, ()>>>,
}

impl AuthConfig {
    pub fn new(username: String, password: String) -> Self {
        Self { username, password, sessions: Arc::new(Mutex::new(HashMap::new())) }
    }

    fn create_session(&self, username: &str, password: &str) -> Option<String> {
        if username == self.username && password == self.password {
            let token = Uuid::new_v4().to_string();
            self.sessions.lock().unwrap().insert(token.clone(), ());
            Some(token)
        } else {
            None
        }
    }

    pub fn is_valid(&self, token: &str) -> bool {
        self.sessions.lock().unwrap().contains_key(token)
    }

    fn revoke(&self, token: &str) {
        self.sessions.lock().unwrap().remove(token);
    }
}

fn session_cookie(headers: &HeaderMap) -> Option<String> {
    headers.get(header::COOKIE)?.to_str().ok().and_then(|c| {
        c.split(';').find_map(|p| p.trim().strip_prefix("session=").map(str::to_owned))
    })
}

// ── Middleware ────────────────────────────────────────────────────────────────

pub async fn require_auth(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let authed = session_cookie(request.headers())
        .as_deref()
        .map(|t| state.auth.is_valid(t))
        .unwrap_or(false);

    if authed {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response()
    }
}

// ── Handlers ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct LoginBody {
    username: String,
    password: String,
}

pub async fn login(State(state): State<AppState>, Json(body): Json<LoginBody>) -> Response {
    match state.auth.create_session(&body.username, &body.password) {
        Some(token) => (
            StatusCode::OK,
            [(
                header::SET_COOKIE,
                format!("session={token}; HttpOnly; Path=/; SameSite=Strict"),
            )],
            Json(serde_json::json!({"ok": true})),
        )
            .into_response(),
        None => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "invalid credentials"})),
        )
            .into_response(),
    }
}

pub async fn logout(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(token) = session_cookie(&headers) {
        state.auth.revoke(&token);
    }
    (
        StatusCode::OK,
        [(
            header::SET_COOKIE,
            "session=; HttpOnly; Path=/; Max-Age=0; SameSite=Strict".to_owned(),
        )],
        Json(serde_json::json!({"ok": true})),
    )
        .into_response()
}

pub async fn check(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let authed = session_cookie(&headers)
        .as_deref()
        .map(|t| state.auth.is_valid(t))
        .unwrap_or(false);
    if authed {
        (StatusCode::OK, Json(serde_json::json!({"ok": true}))).into_response()
    } else {
        (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "unauthorized"}))).into_response()
    }
}
