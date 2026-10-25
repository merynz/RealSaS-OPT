-- +goose Up

-- AXIS V5.4.1 is admitted to the RESEARCH channel only when the exact
-- self-contained runtime package is identified and loadable, the historical
-- FIT1 V5.4.1 terminal result is preserved as evidence, and the runtime
-- authority contract remains hard-causal / no-compiler-reselection / no
-- diffusion-runtime authority.
--
-- This policy deliberately does NOT require or imply FIT8, LOFO, unseen
-- generalization, PRODUCT_CANDIDATE, or PRODUCTION qualification. Those remain
-- separate future exams and channels.
--
-- Canonical policy content (sorted compact JSON + newline) hashed into
-- policy_sha256:
-- {"architecture_id":"RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1","channel":"RESEARCH","family":"AXIS","policy_version":1,"required_evaluation_scopes":["RUNTIME_PACKAGE_IDENTITY_V1","RUNTIME_SELF_CONTAINED_LOAD_V1","HISTORICAL_FIT1_TERMINAL_V541","AXIS_RUNTIME_AUTHORITY_CONTRACT_V1"],"schema":"RealSaS.ModelPromotionPolicy.v1"}
INSERT INTO model_promotion_policies
    (id, family, architecture_id, channel, policy_version,
     required_evaluation_scopes, policy_sha256, sealed_by)
VALUES
    ('9389310f-9555-5a42-ba35-a98f531ca725',
     'AXIS',
     'RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1',
     'RESEARCH',
     1,
     ARRAY[
       'RUNTIME_PACKAGE_IDENTITY_V1',
       'RUNTIME_SELF_CONTAINED_LOAD_V1',
       'HISTORICAL_FIT1_TERMINAL_V541',
       'AXIS_RUNTIME_AUTHORITY_CONTRACT_V1'
     ]::text[],
     'e715f018de5c03f5561989829f70a0cff8eeff0a692b2c73fc67de7300e7cfb9',
     'migration-00020-axis-v541-research-policy');

INSERT INTO model_promotion_policy_bindings
    (family, architecture_id, channel, policy_id, bound_by)
VALUES
    ('AXIS',
     'RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1',
     'RESEARCH',
     '9389310f-9555-5a42-ba35-a98f531ca725',
     'migration-00020-axis-v541-research-policy');

-- +goose Down

DELETE FROM model_promotion_policy_bindings
WHERE family='AXIS'
  AND architecture_id='RealSaS.AXIS41.BFSHardParent.OrderedDiscreteXYZ256.FrozenCausal.v5_4_1'
  AND channel='RESEARCH'
  AND policy_id='9389310f-9555-5a42-ba35-a98f531ca725';

-- The immutable-policy trigger forbids ordinary DELETE. Down migrations are a
-- schema rollback authority operation, so drop/recreate that guard around the
-- exact policy row removal only.
DROP TRIGGER trg_model_promotion_policy_immutable ON model_promotion_policies;
DELETE FROM model_promotion_policies
WHERE id='9389310f-9555-5a42-ba35-a98f531ca725';
CREATE TRIGGER trg_model_promotion_policy_immutable
BEFORE UPDATE OR DELETE ON model_promotion_policies
FOR EACH ROW
EXECUTE FUNCTION enforce_model_promotion_policy_immutable();
