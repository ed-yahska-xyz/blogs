# Two-Factor Auth with Better Auth

Adding TOTP and backup codes to a server-rendered app

Most Better Auth tutorials reach for the client SDK and a React hook. This one
takes the longer way round — wiring two-factor authentication with plain server
endpoints and HTML forms — because doing it by hand is the best way to
understand what the plugin actually does.

We will cover:

1. What TOTP is, and the two flows you have to build (enrollment vs. sign-in)
2. Enabling the plugin and migrating the database
3. Enrollment: turning 2FA on for a user
4. Sign-in: challenging for a code and issuing a session
5. Backup codes — the recovery path people forget
6. Hardening: rate limits and the secret that must not move

The examples use [`better-auth`](https://www.better-auth.com/) `^1.6` with the
Postgres adapter, but nothing here is Postgres-specific.

---

## 1. What you're actually building

**TOTP** (Time-based One-Time Password) is the six-digit code your authenticator
app shows. Server and app share a secret once, at enrollment. After that, both
sides independently compute `HMAC(secret, current_30s_window)` and truncate it to
6 digits. No network round-trip, no SMS — the code matches because the clocks
agree.

The crucial mental model: **2FA is two separate flows, not one.**

```
ENROLLMENT (one time, while already logged in)
  enable ──► scan QR ──► verify first code ──► 2FA is now active

SIGN-IN (every time, after the password)
  password ──► "needs second factor" ──► verify code ──► session issued
```

Better Auth models both, but they use different endpoints and have different
preconditions. Conflating them is the usual source of confusion, so we'll keep
them firmly apart.

---

## 2. Enable the plugin

Add the `twoFactor` plugin to your server auth instance:

```ts
import { betterAuth } from "better-auth";
import { twoFactor } from "better-auth/plugins";

export const auth = betterAuth({
  database: pool, // your adapter
  emailAndPassword: { enabled: true },
  plugins: [
    twoFactor({
      issuer: "your-app.com", // shows up as the account label in the authenticator app
    }),
  ],
});
```

> **TypeScript tip:** if you build the options object separately, use
> `} satisfies BetterAuthOptions` rather than `const config: BetterAuthOptions = {...}`.
> An explicit type annotation erases the plugins tuple, and you lose the
> `auth.api.verifyTOTP` / `enableTwoFactor` methods from the inferred type.

If you use the client SDK, mirror it there:

```ts
import { createAuthClient } from "better-auth/client";
import { twoFactorClient } from "better-auth/client/plugins";

export const authClient = createAuthClient({
  plugins: [twoFactorClient()],
});
```

### The schema

The plugin adds one table and one column:

```sql
alter table "user" add column "twoFactorEnabled" boolean;

create table "twoFactor" (
  "id"          text not null primary key,
  "secret"      text not null,   -- the TOTP secret, ENCRYPTED at rest
  "backupCodes" text not null,   -- encrypted, comma/JSON-packed
  "userId"      text not null references "user" ("id") on delete cascade,
  "verified"    boolean
);
```

Don't hand-write this if you can avoid it — generate it from your config so it
always matches your adapter's conventions:

```bash
bunx @better-auth/cli generate --config path/to/auth.ts
```

Note that `secret` is stored **encrypted**, using your `BETTER_AUTH_SECRET`.
Remember that — it comes back to bite people in section 6.

---

## 3. Enrollment

Enrollment happens while the user is **already logged in**. It's two steps, and
the gap between them matters.

**Step one — generate a secret:**

```ts
// POST /enroll-2fa  (user has a valid session)
const { totpURI, backupCodes } = await auth.api.enableTwoFactor({
  body: { password },          // re-confirm the password
  headers: request.headers,    // the current session
});
```

You get back:
- `totpURI` — an `otpauth://totp/...?secret=...` string. Render it as a QR code
  for the user to scan (or show the `secret` query param for manual entry).
- `backupCodes` — show these **once** and tell the user to save them.

**Step two — confirm the user can produce a code:**

```ts
// POST /enroll-2fa/confirm
await auth.api.verifyTOTP({
  body: { code },              // the 6 digits from their app
  headers: request.headers,
});
```

### The subtlety almost everyone misses

With the default config, `enableTwoFactor` does **not** turn 2FA on. It stores an
*unverified* secret. Two-factor only becomes enforced when that first
`verifyTOTP` succeeds — that's the moment Better Auth flips
`user.twoFactorEnabled = true`.

This is deliberate and good: it stops a user from locking themselves out by
mistyping the QR scan. If they never confirm, nothing changes and they can sign
in with just a password.

```
enableTwoFactor()      → secret saved, verified = false, 2FA NOT enforced
verifyTOTP() succeeds  → verified = true, twoFactorEnabled = true, 2FA enforced
```

(If you *want* one-shot enable without confirmation — e.g. an admin CLI — pass
`skipVerificationOnEnable: true` to the plugin. Use it knowingly.)

---

## 4. Sign-in

Now the part that runs on every login. Here's the server-rendered, no-JavaScript
version, which makes the control flow explicit.

When you call the email sign-in, inspect the response **body**, not just the
status:

```ts
// POST /login
const res = await auth.api.signInEmail({
  body: { email, password },
  asResponse: true,            // we want the raw Response (cookies + body)
});

const setCookies = res.headers.getSetCookie?.() ?? [];
const payload = await res.clone().json().catch(() => null);

if (payload?.twoFactorRedirect) {
  // Password was correct, but the account has 2FA. Better Auth did NOT issue a
  // full session — it set a short-lived "two-factor" cookie instead. Forward
  // that cookie and send the user to the code-entry step.
  const redirect = Response.redirect("/login/2fa", 302);
  for (const c of setCookies) redirect.headers.append("Set-Cookie", c);
  return redirect;
}

// No 2FA on this account: setCookies already contains a full session.
```

Then the second step verifies the code against that two-factor cookie:

```ts
// POST /login/2fa  (carries the two-factor cookie from the previous step)
const res = await auth.api.verifyTOTP({
  body: { code },
  headers: request.headers,    // includes the two-factor cookie
  asResponse: true,
});

if (!res.ok) return renderError("Invalid or expired code.");

// Success: res now carries the real session cookie. Forward it and you're in.
const redirect = Response.redirect("/", 302);
for (const c of res.headers.getSetCookie?.() ?? []) {
  redirect.headers.append("Set-Cookie", c);
}
return redirect;
```

### Why this design is nice

Because a full session is only issued *after* `verifyTOTP`, you get a useful
invariant for free:

> A fully-issued session implies the second factor was completed.

So your route guards don't need any special "did they do 2FA?" check. If
`getSession()` returns a session, 2FA is done. The password-only interim state
simply isn't a session.

---

## 5. Backup codes

Section 3 handed the user backup codes and told them to save the codes. If your
verify step only ever calls `verifyTOTP`, **those codes are decorative** — and a
lost phone means a permanently locked-out account.

Backup codes are a different endpoint, and they're not six digits (Better Auth
generates them as `xxxxx-xxxxx`, alphanumeric). So:

1. Don't constrain the input to digits only. A `pattern="[0-9]*"` on the code
   field silently rejects backup codes.
2. Route by shape: a plain 6-digit string is a TOTP code; anything else is a
   backup code.

```ts
const isTotp = /^\d{6}$/.test(code);

const res = isTotp
  ? await auth.api.verifyTOTP({ body: { code }, headers, asResponse: true })
  : await auth.api.verifyBackupCode({ body: { code }, headers, asResponse: true });
```

Both verify against the same two-factor cookie, so this drops straight into the
sign-in step from section 4. Each backup code works once.

---

## 6. Hardening

Two things that aren't optional in production.

### Rate-limit the verify endpoint

A 6-digit code is a space of 1,000,000, and the two-factor cookie lives for
~10 minutes by default. Without a limit, that's brute-forceable. Put a limiter
on the code-verification route — e.g. 10 attempts per 5 minutes per client IP.

One trap if you're behind a reverse proxy: the proxy *appends* the real client
IP to `X-Forwarded-For`, so the **last** entry is trustworthy and the leftmost is
attacker-controlled. Key your limiter on the last element, or a determined
attacker just rotates the leftmost value to get a fresh bucket:

```ts
// last entry = the IP your proxy observed; leftmost = client-supplied, spoofable
keyGenerator: (req) => req.headers["x-forwarded-for"]?.split(",").pop()?.trim() ?? "global",
```

### Treat `BETTER_AUTH_SECRET` as the master key

Recall from section 2 that the TOTP secret is stored **encrypted** with
`BETTER_AUTH_SECRET`. That has a consequence people discover the hard way:

> If you rotate `BETTER_AUTH_SECRET`, every stored 2FA secret becomes
> undecryptable, and every user's authenticator codes stop verifying.

Rotating that secret already invalidates sessions; with 2FA enabled it also
locks out the second factor. Plan a re-enrollment if you ever rotate it, and make
sure any out-of-band enrollment tooling (a seed script, a one-off container) uses
the **same** secret as the running app — enroll with a different value and every
code will fail at login.

---

## Recap

- 2FA is **two flows**: enrollment (logged-in: `enableTwoFactor` → `verifyTOTP`)
  and sign-in (`signInEmail` → `twoFactorRedirect` → `verifyTOTP`).
- `enableTwoFactor` stores an *unverified* secret; the first successful
  `verifyTOTP` is what actually enforces 2FA.
- A full session is only issued after the second factor, so "has a session"
  already means "passed 2FA."
- Support `verifyBackupCode`, or the backup codes you handed out are a lie.
- Rate-limit verification (keyed on the trustworthy `X-Forwarded-For` entry), and
  never casually rotate `BETTER_AUTH_SECRET` once 2FA is live.

Build it once by hand like this and the client SDK's `twoFactor.*` helpers stop
being magic — they're just these same two flows with the fetch calls written for
you.
