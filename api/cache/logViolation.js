const getLogStores = require('./getLogStores');
const banViolation = require('./banViolation');

/**
 * Logs the violation.
 *
 * @param {Object} req - Express request object containing user information.
 * @param {Object} res - Express response object.
 * @param {string} type - The type of violation.
 * @param {Object} errorMessage - The error message to log.
 * @param {number} [score=1] - The severity of the violation. Defaults to 1
 */
const logViolation = async (req, res, type, errorMessage, score = 1) => {
  const userId = req.user?.id ?? req.user?._id;
  if (!userId) {
    return;
  }
  const logs = getLogStores('general');
  const violationLogs = getLogStores(type);

  const userViolations = (await violationLogs.get(userId)) ?? 0;
  const violationCount = userViolations + score;
  await violationLogs.set(userId, violationCount);

  errorMessage.user_id = userId;
  errorMessage.prev_count = userViolations;
  errorMessage.violation_count = violationCount;
  errorMessage.date = new Date().toISOString();

  await banViolation(req, res, errorMessage);
  const userLogs = (await logs.get(userId)) ?? [];
  userLogs.push(errorMessage);
  await logs.set(userId, userLogs);
};

module.exports = logViolation;
