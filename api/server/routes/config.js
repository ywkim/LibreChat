const express = require('express');
const router = express.Router();
const { isEnabled } = require('../utils');

router.get('/', async function (req, res) {
  try {
    const appTitle = process.env.APP_TITLE || 'LibreChat';
    const googleLoginEnabled = !!process.env.GOOGLE_CLIENT_ID && !!process.env.GOOGLE_CLIENT_SECRET;
    const facebookLoginEnabled =
      !!process.env.FACEBOOK_CLIENT_ID && !!process.env.FACEBOOK_CLIENT_SECRET;
    const openidLoginEnabled =
      !!process.env.OPENID_CLIENT_ID &&
      !!process.env.OPENID_CLIENT_SECRET &&
      !!process.env.OPENID_ISSUER &&
      !!process.env.OPENID_SESSION_SECRET;
    const openidLabel = process.env.OPENID_BUTTON_LABEL || 'Login with OpenID';
    const openidImageUrl = process.env.OPENID_IMAGE_URL;
    const githubLoginEnabled = !!process.env.GITHUB_CLIENT_ID && !!process.env.GITHUB_CLIENT_SECRET;
    const discordLoginEnabled =
      !!process.env.DISCORD_CLIENT_ID && !!process.env.DISCORD_CLIENT_SECRET;
    const serverDomain = process.env.DOMAIN_SERVER || 'http://localhost:3080';
    const registrationEnabled = isEnabled(process.env.ALLOW_REGISTRATION);
    const socialLoginEnabled = isEnabled(process.env.ALLOW_SOCIAL_LOGIN);
    const checkBalance = isEnabled(process.env.CHECK_BALANCE);
    const emailEnabled =
      !!process.env.EMAIL_SERVICE &&
      !!process.env.EMAIL_USERNAME &&
      !!process.env.EMAIL_PASSWORD &&
      !!process.env.EMAIL_FROM;
    const slackInviteEnabled = !!process.env.SLACK_INVITE_URL;
    const slackInviteUrl = process.env.SLACK_INVITE_URL;

    return res.status(200).send({
      appTitle,
      googleLoginEnabled,
      facebookLoginEnabled,
      openidLoginEnabled,
      openidLabel,
      openidImageUrl,
      githubLoginEnabled,
      discordLoginEnabled,
      serverDomain,
      registrationEnabled,
      socialLoginEnabled,
      emailEnabled,
      slackInviteEnabled,
      slackInviteUrl,
      checkBalance,
    });
  } catch (err) {
    console.error(err);
    return res.status(500).send({ error: err.message });
  }
});

module.exports = router;
